
import pdb
import os
import pickle
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time
import math
from nilearn import plotting
import nibabel
from nilearn.input_data import NiftiMasker
import nilearn.masking as masking
from nilearn.image import load_img, resample_img, math_img
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict
from nilearn.datasets import load_mni152_template
from data import constants
import scipy.sparse as scsp
import torch.sparse as tsp


def get_3d_convolution_adjacency(input_volume, kernel_sizes, strides, paddings):
    # input_volume: an (3) dimensional volume, represented as a list.
    # input_volume[i][j][k] = -1 if its not a valid location
    # input_volume[i][j][k] = i if its a valid location
    # output_volume[i'][j'][k'] follows the same convention.
    # adjacency is from valid positions to valid positions
    if not(isinstance(kernel_sizes, list)):
        kernel_sizes = [kernel_sizes, kernel_sizes, kernel_sizes]
    if not(isinstance(strides, list)):
        strides = [strides, strides, strides]
    if not(isinstance(paddings, list)):
        paddings = [paddings, paddings, paddings]

    inD = len(input_volume)
    inH = len(input_volume[0])
    inW = len(input_volume[0][0])
    # Compute new I, J, K
    outD = math.floor(1 + (inD + 2 * paddings[0] - kernel_sizes[0]) / strides[0])
    outH = math.floor(1 + (inH + 2 * paddings[1] - kernel_sizes[1]) / strides[1])
    outW = math.floor(1 + (inW + 2 * paddings[2] - kernel_sizes[2]) / strides[2])

    output_volume = [[[-1 for _ in range(outW)] for _ in range(outH)] for _ in range(outD)]

    adjacency = []
    for d in range(outD):
        for h in range(outH):
            for w in range(outW):
                dependecies = []
                mind_ = max(0, d * strides[0] - kernel_sizes[0] // 2)
                maxd_ = min(inD, 1 + d * strides[0] + kernel_sizes[0] // 2)
                minh_ = max(0, h * strides[1] - kernel_sizes[1] // 2)
                maxh_ = min(inH, 1 + h * strides[1] + kernel_sizes[1] // 2)
                minw_ = max(0, w * strides[2] - kernel_sizes[2] // 2)
                maxw_ = min(inW, 1 + w * strides[2] + kernel_sizes[2] // 2)
                for d_ in range(mind_, maxd_):
                    for h_ in range(minh_, maxh_):
                        for w_ in range(minw_, maxw_):
                            if input_volume[d_][h_][w_] != -1:
                                dependecies.append(input_volume[d_][h_][w_])
                if len(dependecies) > 0:
                    output_volume[d][h][w] = len(adjacency)
                    adjacency.append(dependecies)
    return output_volume, adjacency


def random_graph_adjacency_list(inn, outn, density=0.0002):
    adj = [[i] for i in range(outn)]
    for i in range(inn):
        adj[np.random.randint(0, outn)].append(i)  # Ensure every input has 1 parent.
    for j in range(outn):
        count = 1 + int(abs(np.random.normal(density, density / 2)) * inn)
        adj[j].extend(list(set(np.random.randint(0, outn, size=count).tolist())))
    return adj


def random_tree_adjacency_list(inn, outn):
    adj = [[] for _ in range(outn)]
    for i in range(inn):
        j = np.random.randint(0, outn)
        adj[j].append(i)
    return adj


def make_checkpoint(model, optimizer, epoch):
    chk = {}
    if isinstance(model, nn.DataParallel):
        chk['model'] = model.module.state_dict()
    else:
        chk['model'] = model.state_dict()
    chk['optimizer'] = optimizer.state_dict()
    chk['epoch'] = epoch
    return chk


def kfold_list_split(ls, k):
    # K-fold split
    # returns a list of tuples of lists
    L = len(ls)
    l = L // k
    lsls = []
    for i in range(k):
        start = i * l
        end = start + l
        lsls.append((ls[:start] + ls[end:], ls[start:end]))
    return lsls

def scsp2tsp(mat):
    return tsp.FloatTensor(torch.from_numpy(np.stack([mat.row, mat.col])).long(), torch.from_numpy(mat.data).float(), mat.shape)


def periodic_integer_delta(inp, every=10, start=-1):
    return (inp % every) == ((start + every) % every)


def adj_matrix2adj_list(adj_mat):
    adj_list = [[] for i in range(adj_mat.shape[0])]
    if isinstance(adj_mat, torch.Tensor):
        indices = adj_mat._indices().numpy()
    elif isinstance(scsp.coo_matrix):
        indices = np.stack([adj_mat.row, adj_mat.col])
    else:  # Dense matrix?
        indices = []
        for r in range(adj_mat.shape[0]):
            for c in range(adj_mat.shape[1]):
                if adj_mat[r, c] > 0:
                    indices.append(r, c)
        indices = np.array(indices).T
    indices = indices


def transpose_adj_list(n, m, adj_list):
    # adj_list: n nodes to m nodes
    adjT = [[] for i in range(m)]
    for nidx, nadj in enumerate(adj_list):
        for midx in nadj:
            adjT[midx].append(nidx)
    return adjT


def dump_everything(args):
    # Destination:
    destination_dir = os.path.join(args.base_output, args.run_code)
    destination_file = os.path.join(destination_dir, "information.json")
    obj = {}
    args_serializable = {k: v for k, v in args.__dict__.items() if ((k != "meta") and (k != "wtree"))}
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
        yield ans


def save_images(tensor_list, title_list, filename, mu_=0.0, std_=1.0, figsize=(10, 10), nrows=2, ncols=1, indexes=None, cut_coords=None, show_instead_of_save=False):
    figure = plt.figure(figsize=figsize)
    for idx, tensor in enumerate(tensor_list):
        ax = plt.subplot(nrows, ncols, indexes[idx] if indexes else idx)
        if tensor.shape[0] == 91:
            mask = constants.original_brain_mask
        else:
            mask = constants.downsampled_brain_mask
        plotting.plot_stat_map(
            math_img("img1 * img2",
                img1=nibabel.Nifti1Image(
                    mu_ + std_ * tensor.detach().cpu().numpy().squeeze(), mask.affine
                ),
                img2=mask
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
