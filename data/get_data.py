"""
Script to download and split all datasets
"""

import pdb
import pickle
import random
import os
import argparse
import json
import nibabel
import numpy as np
import nilearn
import nilearn.image
import nibabel
import nilearn.datasets as nv
import pandas as pd
import copy
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.image import load_img, resample_img
import time
from data.constants import nv_ids, hcp_subject_list, brain_mask


def get_range(image_file_list, dim=3):
    min_, max_ = None, None
    n = 0
    shape = None

    # Performs two pass algorithm to get mu and std.
    xn_bar = 0.0
    mean_image = None
    start = time.time()
    for image_file in image_file_list:
        img = nibabel.load(image_file)
        data = np.nan_to_num(img.get_data(), copy=False)
        if dim:
            n_elems = data.shape[dim]
        else:
            n_elems = 1
        n += n_elems
        if shape is None:
            shape = list(copy.deepcopy(data.shape))
            if dim:
                del shape[dim]

        if min_:
            min_ = min(min_, data.min())
        else:
            min_ = data.min()

        if max_:
            max_ = max(max_, data.max())
        else:
            max_ = data.max()

        temp = (data.mean() - xn_bar)
        xn_bar += n_elems * temp / n

        if dim:
            temp_image = data.mean(dim)
        else:
            temp_image = data

        if mean_image is not None:
            mean_image += n_elems * (temp_image - mean_image) / n
        else:
            mean_image = n_elems * temp_image / n
    print("Computed mean: {}s".format(time.time() - start))
    start = time.time()
    mu = xn_bar  # Correct.
    var_image = None
    n = 0
    x2n_bar = 0.0
    for image_file in image_file_list:
        img = nibabel.load(image_file)
        # Subtract mean.
        data = (np.nan_to_num(img.get_data(), copy=False) - xn_bar)
        if dim:
            n_elems = data.shape[dim]
        else:
            n_elems = 1
        n += n_elems

        temp = (data**2).mean() - x2n_bar
        x2n_bar += n_elems * temp / n
        if dim:
            temp_image = (data**2).mean(dim)
        else:
            temp_image = (data**2)
        if var_image is not None:
            var_image += n_elems * (temp_image - var_image) / n
        else:
            var_image = n_elems * (temp_image) / n
    print("Computer variance in {}s".format(time.time() - start))
    std = np.sqrt(x2n_bar)  # Note: the constant factor doesn't matter for our use.
    std_image = np.sqrt(var_image)
    return min_, max_, n, shape, mu, std, mean_image, std_image


def _assemble(images, images_meta, study):
    records = []
    for image, meta in zip(images, images_meta):
        if study == 'brainpedia':
            this_study = meta['study']
            subject = meta['name'].split('_')[-1]
            contrast = '_'.join(meta['task'].split('_')[1:])
            task = meta['task'].split('_')[0]
        elif study == 'rfmri':
            this_study = study
            # subject, contrast, task
            subject = meta['subject']
            contrast = meta['contrast']
            task = meta['task']
        else:
            this_study = study
            subject = meta['name'].split('_')[0]
            contrast = meta['contrast_definition']
            task = meta['task']
        records.append([image, this_study, subject, task, contrast])
    df = pd.DataFrame(records, columns=['z_map', 'study', 'subject', 'task', 'contrast'])
    return df


def get_rfmri_bunch(hcp_dir):
    data = {
        'images': [os.path.join(hcp_dir, x) for x in os.listdir(hcp_dir)],
        'images_meta': [],
    }
    for img_file in data['images']:
        img_name = os.path.basename(img_file)
        toks = img_name.split("_")
        subject_id = toks[2]
        n_direction_thing = toks[3]
        data['images_meta'].append({
            'contrast': "rfmri",
            'task': "rfmri",
            'subject': str(subject_id),
        })
    return data


def fetch_contrasts(studies: str or List[str] = 'all', data_dir='/data/', rfmri_dir="/data/hcp/downsampled"):
    dfs = []
    if studies[0] == 'all':
        studies = nv_ids.keys()
    print('studies={}'.format(studies))
    for study in studies:
        if study == "rfmri":
            continue  # We don't need rfmri for now.
            data = get_rfmri_bunch(rfmri_dir)
        else:
            if study not in nv_ids:
                return ValueError('Wrong dataset.')
            data = nv.fetch_neurovault_ids([nv_ids[study]], data_dir=data_dir, verbose=10,
                                        mode='download_new')
        dfs.append(_assemble(data['images'], data['images_meta'], study))
    return pd.concat(dfs)


def get_statistics(df):
    df.set_index(keys=['z_map'], drop=False, inplace=True)
    records = []
    # Go over each record in df
    studies = df['study'].unique().tolist()
    for study in studies:
        study_df = df.loc[df.study == study]
        min_, max_, n, shape, mu, std, mean_image, std_image = get_range(study_df.z_map.tolist(), dim=None)
        records.append([study, mu, std, n, mean_image, std_image])
    return pd.DataFrame(records, columns=['study', 'mu', 'sigma', 'n', 'mean_image', 'std_image'])


def resample_all(df):
    df = df.copy()
    new_z_maps = []
    n = df.shape[0]
    for idx in range(n):
        print("{}/{}".format(idx, n))
        image_file = df.iloc[idx].z_map
        if df.iloc[idx].study == "rfmri":
            new_z_maps.append(image_file)
            continue
        img = nibabel.load(image_file)
        # print(img.get_data().shape)
        # exit()
        data = np.nan_to_num(img.get_data(), copy=False).astype(np.float)
        affine = img.affine
        this_data = nilearn.image.resample_to_img(
            nibabel.Nifti1Image(
                data, affine,
            ),
            brain_mask
        )
        this_data = nilearn.image.math_img("img1 * img2", img1=this_data, img2=brain_mask)
        new_image_file = image_file.replace('.nii.gz', '_down.nii.gz')
        this_data.to_filename(new_image_file)
        new_z_maps.append(new_image_file)
    df.z_map = pd.Series(new_z_maps, index=df.index)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs='+', type=str, choices=["all"] + list(nv_ids.keys()), help="Which data")
    parser.add_argument("--fetch", action='store_true', help="Provide option if data doesnt exist")
    parser.add_argument("--resample", default=False, action='store_true', help="resample the data or not")
    parser.add_argument("--stats", default=False, action='store_true', help="Use if data is already downloaded and you just want to get some statistics")
    parser.add_argument("--resampled_stats", default=False, action='store_true', help="stats for the resampled data alone")
    args = parser.parse_args()

    if args.fetch:
        df = fetch_contrasts(args.names)
        df.to_csv("/data/neurovault/dataframe.csv")
    else:
        df = pd.read_csv("/data/neurovault/dataframe.csv")
    # note: doesnt handle all names and stuff
    stats_file_name = "stats"
    if args.resample:
        df = resample_all(df)
        df.to_csv("/data/neurovault/resampled_dataframe.csv")
    elif args.resampled_stats:
        df = pd.read_csv("/data/neurovault/resampled_dataframe.csv")
    else:
        stats_file_name = "original_stats"

    if args.stats:
        # pdb.set_trace()
        stats_df = get_statistics(df)
        stats_df.to_pickle("/data/neurovault/{}.pkl".format(stats_file_name))
        stats_df.to_csv("/data/neurovault/{}.csv".format(stats_file_name))

    # if args.concat_rfmri:
    #     df = df.loc[df.study != "rfmri"]
    #     df.set_index(keys=['z_map'], drop=False, inplace=True)
    #     df_new = fetch_contrasts(["rfmri"])
    #     df_new.set_index(keys=['z_map'], drop=False, inplace=True)
    #
    #     stats_df_old = pd.read_csv("/data/neurovault/stats.csv")
    #     stats_df_old = stats_df_old.loc[stats_df_old.study != "rfmri"]
    #     stats_df_new = get_statistics(df_new)
    #     df = pd.concat([df, df_new], sort=False)
    #     stats_df = pd.concat([stats_df_new, stats_df_old], sort=False)
    #
    #     df.to_csv("/data/neurovault/resampled_dataframe.csv")
    #     stats_df.to_pickle("/data/neurovault/stats.pkl")
    #     stats_df.to_csv("/data/neurovault/stats.csv")
    # else:
    #     stats_df = get_statistics(df)
        # Need to compute statistics
