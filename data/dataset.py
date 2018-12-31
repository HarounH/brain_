"""
Handlers for dataset
"""


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
    brain_mask,
    dataframe_csv_file,
    statistics_pkl,
    brain_mask_numpy,
)
from utils.utils import infinite_iter, multi_key_infinite_iter
warnings.simplefilter("ignore")


class Dataset(TorchDataset):
    def __init__(self, df, s, meta, masked=False):
        self.df = df
        self.meta = meta
        self.mu = meta['s2mu'][s].astype(np.float32)  # mean_image
        self.std = meta['s2std'][s].astype(np.float32)  # std_image
        self.n = self.df.shape[0]
        self.masked = masked
        if masked:
            self.mu = masking.apply_mask(nibabel.Nifti1Image(self.mu, brain_mask.affine), brain_mask)
            self.std = masking.apply_mask(nibabel.Nifti1Image(self.std, brain_mask.affine), brain_mask)
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        loc = self.df.iloc[idx]
        image_file = loc.z_map
        study = self.meta['s2i'][loc.study]
        task = self.meta['t2i'][loc.task]
        contrast = self.meta['c2i'][loc.contrast]
        img = nibabel.load(image_file)
        if self.masked:
            this_data = masking.apply_mask(img, brain_mask).astype(np.float32)
            this_data = (this_data - self.mu) / (1e-4 + self.std)
            this_data = (-1.0 + 2.0 * (this_data - this_data.min()) / (this_data.max() - this_data.min()))  # [-1 -> 1]
        else:
            this_data = np.nan_to_num(img.get_data(), copy=False).astype(np.float32)
            this_data = (this_data - self.mu) / (1e-4 + self.std)
            this_data = brain_mask_numpy * (-1.0 + 2.0 * (this_data - this_data.min()) / (this_data.max() - this_data.min()))  # [-1 -> 1]
        return this_data, study, task, contrast


def get_datasets(studies, subject_split, debug, masked=False):
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
            study_df = study_df.sort_values(by='subject', axis=0, inplace=False)
            # set the index to be this and don't drop
            study_df = study_df.set_index(keys=['subject'], drop=False, inplace=False)
            subjects = study_df['subject'].unique().tolist()
            train_subjects = subjects[:len(subjects) // 2]
            test_subjects = subjects[len(subjects) // 2:]  # Split by subject.
            train_dfs = [study_df.loc[study_df.subject == subj] for subj in train_subjects]
            test_dfs = [study_df.loc[study_df.subject == subj] for subj in test_subjects]
            train_dfs_by_study[study] = train_dfs
            test_dfs_by_study[study] = test_dfs
        else:
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
        training_datasets[s] = Dataset(pd.concat(dfs), s, meta, masked)
    for s, dfs in test_dfs_by_study.items():
        testing_datasets[s] = Dataset(pd.concat(dfs), s, meta, masked)

    return training_datasets, testing_datasets, meta


def get_dataloaders(studies, subject_split, debug, batch_size, num_workers, masked=False):
    # Get datasets
    training_datasets, testing_datasets, meta = get_datasets(studies, subject_split, debug, masked)
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
