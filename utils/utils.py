
import pdb
import os
import pickle
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
plt.switch_backend('agg')
from nilearn import plotting
import nibabel
from nilearn.input_data import NiftiMasker
import nilearn.masking as masking
from nilearn.image import load_img, resample_img, math_img
import torch
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict
from nilearn.datasets import load_mni152_template
from data import constants

def periodic_integer_delta(inp, every=10, start=-1):
    return (inp % every) == ((start + every) % every)


def dump_everything(args):
    # Destination:
    destination_dir = os.path.join(args.base_output, args.run_code)
    destination_file = os.path.join(destination_dir, "information.json")
    obj = {}
    args_serializable = {k: v for k, v in args.__dict__.items() if k != "meta"}
    args_serializable["meta"] = {k: v for k, v in args.__dict__["meta"].items() if ((k != "s2mu") and (k != "s2std"))}
    args_serializable["meta"]["s2mu"] = {k: v.tolist() for k, v in args.__dict__["meta"]["s2mu"].items()}
    args_serializable["meta"]["s2std"] = {k: v.tolist() for k, v in args.__dict__["meta"]["s2std"].items()}
    obj["args"] = args_serializable
    with open(destination_file, 'w') as f:
        json.dump(obj, f, indent="\t")


def infinite_iter(iterable):
    # stolen from https://github.com/arthurmensch/cogspaces/blob/master/cogspaces/data.py
    while True:
        for elem in iterable:
            yield elem


def multi_key_infinite_iter(iters):
    # iters is a dict from key->iterable
    infinite_iters = {k: iter(infinite_iter(x)) for k, x in iters.items()}
    while True:
        temp_time = time.time()
        ans = {k: next(loader) for k, loader in infinite_iters.items()}
        print("Loading took {}".format(time.time() - temp_time))
        yield ans


def save_images(tensor_list, title_list, filename, mu_=0.0, std_=1.0, figsize=(10, 10), nrows=2, ncols=1, indexes=None, cut_coords=None, show_instead_of_save=False):
    figure = plt.figure(figsize=figsize)
    for idx, tensor in enumerate(tensor_list):
        ax = plt.subplot(nrows, ncols, indexes[idx] if indexes else idx)
        plotting.plot_stat_map(
            math_img("img1 * img2",
                img1=nibabel.Nifti1Image(
                    mu_ + std_ * tensor.detach().cpu().numpy().squeeze(), constants.brain_mask.affine
                ),
                img2=constants.brain_mask
            ),
            axes=ax,
            title=title_list[idx],
            cut_coords=cut_coords,
            # bg_img=downsampled_template,
            threshold=False,
            black_bg=False,
        )
    if show_instead_of_save:
        plt.plot()
    else:
        figure.savefig(filename)
        plt.close()
