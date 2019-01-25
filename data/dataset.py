"""
Handlers for dataset
"""


import math
import pandas as pd
import pdb
import os
import pickle
import json
from collections import deque, defaultdict
import nilearn
import nilearn.image
import nibabel
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
import scipy.ndimage
from nilearn.image import load_img, resample_img
import nilearn.masking as masking
from nilearn.datasets import load_mni152_template
import copy
import torch
import warnings
import time
from data.constants import (
    nv_ids,
    downsampled_brain_mask,
    original_brain_mask,
    downsampled_dataframe_csv_file,
    original_dataframe_csv_file,
    downsampled_statistics_pkl,
    original_statistics_pkl,
    downsampled_brain_mask_numpy,
    original_brain_mask_numpy,
)
from utils.utils import (
    infinite_iter,
    multi_key_infinite_iter,
    kfold_list_split,

)
warnings.simplefilter("ignore")


class Dataset(TorchDataset):
    def __init__(self, df, s, meta, masked=False, downsampled=False, normalization='none', not_lazy=False):
        '''
        normalization:
            none: Do no normalization
            11: Normalize to [-1, 1] without mean and center normalization
            0c: 0 center data but dont -1, 1 normalize
            both: perform both types of normalization
        '''
        assert normalization in ['none', '11', '0c', 'both']
        self.df = df
        self.meta = meta
        self.mu = meta['s2mu'][s].astype(np.float32)  # mean_image
        self.std = meta['s2std'][s].astype(np.float32)  # std_image
        self.n = self.df.shape[0]

        self.masked = masked
        self.downsampled = downsampled
        self.normalization = normalization
        self.brain_mask = brain_mask = downsampled_brain_mask if downsampled else original_brain_mask
        self.brain_mask_numpy = downsampled_brain_mask_numpy if downsampled else original_brain_mask_numpy

        if masked:
            self.mu = masking.apply_mask(nibabel.Nifti1Image(self.mu, self.brain_mask.affine), self.brain_mask)
            self.std = masking.apply_mask(nibabel.Nifti1Image(self.std, self.brain_mask.affine), self.brain_mask)
            self.mul_mask = 1.0
        else:
            self.mul_mask = self.brain_mask_numpy

        if normalization == 'none':
            def nf_(x):
                return x
        elif normalization == 'both':
            def nf_(x):
                temp = (x - self.mu) / (1e-4 + self.std)
                return self.mul_mask * (-1.0 + 2.0 * (temp - temp.min()) / (temp.max() - temp.min()))
        elif normalization == '0c':
            def nf_(x):
                return (x - self.mu) / (1e-4 + self.std)
        elif normalization == '11':
            def nf_(temp):
                return self.mul_mask * (-1.0 + 2.0 * (temp - temp.min()) / (temp.max() - temp.min()))

        self.normalization_func = nf_

        self.not_lazy = not_lazy
        if not_lazy:
            def load_image(filename):
                img = nibabel.load(image_file)
                if self.masked:
                    this_data = masking.apply_mask(img, self.brain_mask).astype(np.float32)
                else:
                    this_data = np.nan_to_num(img.get_data(), copy=False).astype(np.float32)
                return this_data
            self.data = np.stack([self.normalization_func(load_img(self.df.iloc[idx])) for i in range(self.n)])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        loc = self.df.iloc[idx]
        image_file = loc.z_map
        study = self.meta['s2i'][loc.study]
        task = self.meta['t2i'][loc.task]
        contrast = self.meta['c2i'][loc.contrast]
        if self.not_lazy:
            return self.data[idx], study, task, contast

        img = nibabel.load(image_file)

        if self.masked:
            this_data = masking.apply_mask(img, self.brain_mask).astype(np.float32)
        else:
            this_data = np.nan_to_num(img.get_data(), copy=False).astype(np.float32)

        return self.normalization_func(this_data), study, task, contrast


def get_datasets(studies, subject_split, debug, masked=False, downsampled=False, normalization='none', not_lazy=False):
    if not(downsampled):
        dataframe_csv_file = original_dataframe_csv_file
        statistics_pkl = original_statistics_pkl
    else:
        dataframe_csv_file = downsampled_dataframe_csv_file
        statistics_pkl = downsampled_statistics_pkl

    df = pd.read_csv(dataframe_csv_file)
    stats = pd.read_pickle(statistics_pkl)
    stats.set_index(keys=['study'], drop=False, inplace=True)
    stats_dict = stats.to_dict(orient='index')
    studies = [x for x in df.study.unique().tolist() if x in studies]
    print("Loading studies: {}".format(studies))
    # Meta object used to share dataset labelling information between datasets.
    meta = {}
    meta['s2i'] = {}
    meta['t2i'] = {}
    meta['c2i'] = {}
    meta['i2s'] = {}
    meta['i2t'] = {}
    meta['i2c'] = {}
    meta['si2ti'] = defaultdict(lambda: [])
    meta['ti2ci'] = defaultdict(lambda: [])
    meta['s2mu'] = {}
    meta['s2std'] = {}

    train_dfs_by_study = {}
    test_dfs_by_study = {}

    for study in studies:
        meta['s2i'][study] = len(meta['s2i'])
        meta['i2s'][meta['s2i'][study]] = study
        study_df = df.loc[df.study == study]

        si = meta['s2i'][study]

        # Split the study_df into half by subject.
        if subject_split:
            print("Splitting by subjects")
            study_df = study_df.sort_values(by='subject', axis=0, inplace=False)
            # set the index to be this and don't drop
            study_df = study_df.set_index(keys=['subject'], drop=False, inplace=False)
            subjects = np.random.permutation(study_df['subject'].unique().tolist())
            train_subjects = subjects[:len(subjects) // 2]
            test_subjects = subjects[len(subjects) // 2:]  # Split by subject.
            train_dfs = [study_df.loc[study_df.subject == subj] for subj in train_subjects]
            test_dfs = [study_df.loc[study_df.subject == subj] for subj in test_subjects]
            train_dfs_by_study[study] = train_dfs
            test_dfs_by_study[study] = test_dfs
        else:
            print("Splitting randomly - not by subject")
            study_df.set_index(keys=['z_map'], drop=False, inplace=True)
            study_df = study_df.reindex(np.random.permutation(study_df.index))
            # Take study_df and equally split it.
            train_df = study_df.iloc[:study_df.shape[0] // 2]
            test_df = study_df.iloc[study_df.shape[0] // 2:]
            train_dfs_by_study[study] = [train_df]
            test_dfs_by_study[study] = [test_df]
        tasks_arr = study_df['task'].unique().tolist()
        for task in tasks_arr:
            if task not in meta['t2i']:
                meta['t2i'][task] = len(meta['t2i'])
                meta['i2t'][meta['t2i'][task]] = task
            ti = meta['t2i'][task]
            meta['si2ti'][si].append(ti)
            contrast_list = study_df.loc[study_df.task == task].contrast.unique().tolist()

            for contrast in contrast_list:
                if contrast not in meta['c2i']:
                    meta['c2i'][contrast] = len(meta['c2i'])
                    meta['i2c'][meta['c2i'][contrast]] = contrast
                ci = meta['c2i'][contrast]
                meta['ti2ci'][ti].append(ci)
        # print("For {} we have tasks:{}".format(study, tasks_arr))
        meta['s2mu'][study] = stats_dict[study]['mean_image']
        meta['s2std'][study] = stats_dict[study]['std_image']

    # Convert them into dicts to keep them serializable
    meta['si2ti'] = dict(meta['si2ti'])
    meta['ti2ci'] = dict(meta['ti2ci'])

    training_datasets, testing_datasets = {}, {}
    for s, dfs in train_dfs_by_study.items():
        training_datasets[s] = Dataset(pd.concat(dfs), s, meta, masked=masked, downsampled=downsampled, normalization=normalization, not_lazy=not_lazy)
    for s, dfs in test_dfs_by_study.items():
        testing_datasets[s] = Dataset(pd.concat(dfs), s, meta, masked=masked, downsampled=downsampled, normalization=normalization, not_lazy=not_lazy)

    return training_datasets, testing_datasets, meta


def get_dataloaders(studies, subject_split, debug, batch_size, num_workers, masked=False, downsampled=False, normalization='none', not_lazy=False):
    # Get datasets
    training_datasets, testing_datasets, meta = get_datasets(studies, subject_split, debug, masked=masked, downsampled=downsampled, normalization=normalization, not_lazy=not_lazy)
    train_loaders = {
        s: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ) for s, dataset in training_datasets.items()
    }
    test_loaders = {
        s: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ) for s, dataset in testing_datasets.items()
    }
    return training_datasets, testing_datasets, meta, train_loaders, test_loaders


def random_splits(ls, k, frac):
    # Split ls k times, with each split being random
    lsls = []
    for i in range(k):
        start = np.random.randint(0, math.floor(len(ls) * (1 - frac)))
        end = start + math.floor(len(ls) * (frac))
        lsls.append((ls[:start] + ls[end:], ls[start:end]))
    return lsls


"""
For good experimentation, we need to do:
for (train_val, test) in comprehensive_splits_of_full_data:
    for train, val in comprehensive_splits_of_train_val:
        fit on train such that val acc is maximized (early stopping)
        measure acc on test
"""
def get_splits(study, outer_k, inner_k, seed, random_outer=None, random_inner=0.2, masked=False, downsampled=False, normalization='none', not_lazy=False):
    # if random_inner is a float, that much fraction is used for validation,
    # if its None, then inner_k equal splits are made.
    # If random_outer is None, then mutually exclusive outer_k splits are made.
    # If random_outer is a float, then outer_k random outer splits are made.
    # assert(outer_k > 1 and inner_k > 1)
    if not(downsampled):
        dataframe_csv_file = original_dataframe_csv_file
        statistics_pkl = original_statistics_pkl
    else:
        dataframe_csv_file = downsampled_dataframe_csv_file
        statistics_pkl = downsampled_statistics_pkl

    df = pd.read_csv(dataframe_csv_file)
    stats = pd.read_pickle(statistics_pkl)
    stats.set_index(keys=['study'], drop=False, inplace=True)
    stats_dict = stats.to_dict(orient='index')
    df = df.loc[df.study == study]
    meta = {}
    meta['s2i'] = {}
    meta['t2i'] = {}
    meta['c2i'] = {}
    meta['i2s'] = {}
    meta['i2t'] = {}
    meta['i2c'] = {}
    meta['si2ti'] = defaultdict(lambda: [])
    meta['ti2ci'] = defaultdict(lambda: [])
    meta['s2mu'] = {}
    meta['s2std'] = {}

    meta['s2i'][study] = len(meta['s2i'])
    meta['i2s'][meta['s2i'][study]] = study
    study_df = df.loc[df.study == study]
    si = meta['s2i'][study]
    tasks_arr = study_df['task'].unique().tolist()
    for task in tasks_arr:
        if task not in meta['t2i']:
            meta['t2i'][task] = len(meta['t2i'])
            meta['i2t'][meta['t2i'][task]] = task
        ti = meta['t2i'][task]
        meta['si2ti'][si].append(ti)
        contrast_list = study_df.loc[study_df.task == task].contrast.unique().tolist()

        for contrast in contrast_list:
            if contrast not in meta['c2i']:
                meta['c2i'][contrast] = len(meta['c2i'])
                meta['i2c'][meta['c2i'][contrast]] = contrast
            ci = meta['c2i'][contrast]
            meta['ti2ci'][ti].append(ci)
    # print("For {} we have tasks:{}".format(study, tasks_arr))
    meta['s2mu'][study] = stats_dict[study]['mean_image']
    meta['s2std'][study] = stats_dict[study]['std_image']

    # Convert them into dicts to keep them serializable
    meta['si2ti'] = dict(meta['si2ti'])
    meta['ti2ci'] = dict(meta['ti2ci'])

    def subject_list_to_dataset(subjects):
        subdf = df[df['subject'].isin(subjects)]
        return Dataset(subdf, study, meta, masked=masked, downsampled=downsampled, normalization=normalization, not_lazy=not_lazy)

    np.random.seed(seed)  # Ensures same split.
    subjects = np.random.permutation(df.subject.unique().tolist()).tolist()
    if random_outer is None:
        outer_subject_splits = kfold_list_split(subjects, outer_k)
    else:
        outer_subject_splits = random_splits(subjects, outer_k, random_outer)
    subject_splits = []
    splits = []
    for (train_val_subjects, test_subjects) in outer_subject_splits:
        if random_inner is None:
            inner_subject_splits = kfold_list_split(train_val_subjects, inner_k)
        else:
            inner_subject_splits = random_splits(train_val_subjects, inner_k, random_inner)
        test_dset = subject_list_to_dataset(test_subjects)
        inner_dsets  = []
        for (train_subjects, val_subjects) in inner_subject_splits:
            train_dset = subject_list_to_dataset(train_subjects)
            val_dset = subject_list_to_dataset(val_subjects)
            inner_dsets.append({'train': train_dset, 'val': val_dset,})
        splits.append({'test': test_dset, 'splits': inner_dsets})
    return splits, meta


if __name__ == '__main__':
    splits, meta = get_splits('archi', 5, 5, 666)
    import pdb; pdb.set_trace()
