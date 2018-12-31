"""
Generator using FGL
"""


import os
import json
import argparse
import numpy as np
import scipy
import random
from random import shuffle
from time import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import torch
from torch import autograd
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from tensorboardX import SummaryWriter
from torch.nn.utils import weight_norm, spectral_norm
from torch.utils.data.dataset import Dataset
from gcn.modules import fgl
import utils.utils as utils
from data import constants


class GeneratorHierarchical0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(GeneratorHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.content_channels = content_channels
        self.z_size = z_size

        #############
        # Linear layers
        #############

        self.study_embedding = weight_norm(nn.Embedding(len(meta['s2i']), content_channels))
        self.task_embedding = weight_norm(nn.Embedding(len(meta['t2i']), content_channels))
        self.contrast_embedding = weight_norm(nn.Embedding(len(meta['c2i']), content_channels))

        self.zfc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
        )

        self.fcs = nn.ModuleList([
            nn.Sequential(nn.Linear(content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(2 * content_channels, content_channels),nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
        ])

        self.node_sizes = [constants.masked_nnz, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        self.channel_sizes = [1, content_channels + (z_size // 16), content_channels + (z_size // 8), content_channels + (z_size // 4), content_channels + (z_size // 2), content_channels + z_size]

        adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.nodes_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from 67615->32768...->128
        # we need to transpose each one and them reverse the list
        adj_list = [utils.transpose_adj_list(self.node_sizes[i], self.node_sizes[i + 1], al) for i, al in enumerate(adj_list)]
        adj_list = adj_list[::-1]

        self.upsample0 = fgl.FGL(self.channel_sizes[-1], self.node_sizes[-1], self.channel_sizes[-2], self.node_sizes[-2], adj_list[0])
        self.upsample1 = fgl.FGL(self.channel_sizes[-2], self.node_sizes[-2], self.channel_sizes[-3], self.node_sizes[-3], adj_list[1])
        self.upsample2 = fgl.FGL(self.channel_sizes[-3], self.node_sizes[-3], self.channel_sizes[-4], self.node_sizes[-4], adj_list[2])
        self.upsample3 = fgl.FGL(self.channel_sizes[-4], self.node_sizes[-4], self.channel_sizes[-5], self.node_sizes[5], adj_list[3])
        self.upsample4 = fgl.FGL(self.channel_sizes[-5], self.node_sizes[5], self.channel_sizes[0], self.node_sizes[0], adj_list[4])

        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-2]))
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-3]))
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-4]))
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-5]))
        self.activation4 = nn.Sequential(nn.Tanh())

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, z, svec, tvec, cvec):
        # z: N, inc
        N = z.shape[0]
        se = self.study_embedding(studies)
        te = self.task_embedding(tasks)
        ce = self.contrast_embedding(contrasts)
        contents = [
            se,
            torch.cat([se, te], dim=1),  # se + te,
            torch.cat([se, te, ce], dim=1),  # se + te + ce
        ]
        contents.append(contents[-1])
        contents.append(contents[-1])  # append twice.
        contents = [self.fcs[i](content) for i, content in enumerate(contents)]
        cur_z = z.unsqueeze(2).expand(N, self.channel_sizes[-1], self.node_sizes[-1])
        for i in range(0, 5):
            upsample = getattr(self, 'upsample{}'.format(i))
            content = contents[i].unsqueeze(2).expand(z.shape[0], self.content_channels, cur_z.shape[2])
            cur_z = upsample(torch.cat([cur_z, content], dim=1))
            if hasattr(self, 'residual{}'.format(i)):
                cur_z = getattr(self, 'residual{}'.format(i))(cur_z)
            if hasattr(self, 'activation{}'.format(i)):
                cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        retval = cur_z[:, 0, ...]
        return retval


class GeneratorHierarchicalRegionwise0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(GeneratorHierarchicalRegionwise0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.content_channels = content_channels
        self.z_size = z_size

        #############
        # Linear layers
        #############

        self.study_embedding = weight_norm(nn.Embedding(len(meta['s2i']), content_channels))
        self.task_embedding = weight_norm(nn.Embedding(len(meta['t2i']), content_channels))
        self.contrast_embedding = weight_norm(nn.Embedding(len(meta['c2i']), content_channels))

        self.zfc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
        )

        self.fcs = nn.ModuleList([
            nn.Sequential(nn.Linear(content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(2 * content_channels, content_channels),nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
            nn.Sequential(nn.Linear(3 * content_channels, content_channels), nn.Dropout(dropout_rate),),
        ])

        self.node_sizes = [constants.masked_nnz, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        self.channel_sizes = [1, content_channels + (z_size // 16), content_channels + (z_size // 8), content_channels + (z_size // 4), content_channels + (z_size // 2), content_channels + z_size]

        list_of_dict_adj_list = []
        cur_level = {-1: wtree.get_leaves()}
        for next_count in self.nodes_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level[-1], next_count)
            list_of_dict_adj_list.append(adj)
        # adj_list contains adj list from 67615->32768...->128
        # we need to transpose each one and them reverse the list
        for i in range(len(list_of_dict_adj_list)):
            list_of_dict_adj_list[i] = {
                k: utils.transpose_adj_list(self.node_sizes[i], self.node_sizes[i + 1], v) for k, v in list_of_dict_adj_list[i].items()
            }
        list_of_dict_adj_list = list_of_dict_adj_list[::-1]

        self.upsample0 = fgl.RegionFGL(self.channel_sizes[-1], self.node_sizes[-1], self.channel_sizes[-2], self.node_sizes[-2], list_of_dict_adj_list[0], reduction='sum')
        self.upsample1 = fgl.RegionFGL(self.channel_sizes[-2], self.node_sizes[-2], self.channel_sizes[-3], self.node_sizes[-3], list_of_dict_adj_list[1], reduction='sum')
        self.upsample2 = fgl.RegionFGL(self.channel_sizes[-3], self.node_sizes[-3], self.channel_sizes[-4], self.node_sizes[-4], list_of_dict_adj_list[2], reduction='sum')
        self.upsample3 = fgl.RegionFGL(self.channel_sizes[-4], self.node_sizes[-4], self.channel_sizes[-5], self.node_sizes[5], list_of_dict_adj_list[3], reduction='sum')
        self.upsample4 = fgl.RegionFGL(self.channel_sizes[-5], self.node_sizes[5], self.channel_sizes[0], self.node_sizes[0], list_of_dict_adj_list[4], reduction='sum')

        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-2]))
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-3]))
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-4]))
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm1d(self.channel_sizes[-5]))
        self.activation4 = nn.Sequential(nn.Tanh())

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, z, svec, tvec, cvec):
        # z: N, inc
        N = z.shape[0]
        se = self.study_embedding(studies)
        te = self.task_embedding(tasks)
        ce = self.contrast_embedding(contrasts)
        contents = [
            se,
            torch.cat([se, te], dim=1),  # se + te,
            torch.cat([se, te, ce], dim=1),  # se + te + ce
        ]
        contents.append(contents[-1])
        contents.append(contents[-1])  # append twice.
        contents = [self.fcs[i](content) for i, content in enumerate(contents)]
        cur_z = z.unsqueeze(2).expand(N, self.channel_sizes[-1], self.node_sizes[-1])
        for i in range(0, 5):
            upsample = getattr(self, 'upsample{}'.format(i))
            content = contents[i].unsqueeze(2).expand(z.shape[0], self.content_channels, cur_z.shape[2])
            cur_z = upsample(torch.cat([cur_z, content], dim=1))
            if hasattr(self, 'residual{}'.format(i)):
                cur_z = getattr(self, 'residual{}'.format(i))(cur_z)
            if hasattr(self, 'activation{}'.format(i)):
                cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        retval = cur_z[:, 0, ...]
        return retval

versions = {
    '0': GeneratorHierarchical0,
}



if __name__ == '__main__':
    # UNIT TEST
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    import data.constants as constants
    args = dotdict()
    args.device = 'cpu'
    args.meta = {}
    args.meta['s2i'] = {"sa": 0, "sb": 1,}
    args.meta['t2i'] = {"taa": 0, "tab": 1, "tba": 2, "tbb": 3, "tbc": 4}
    args.meta['c2i'] = {"caaa": 0, "caab": 1, "caba": 2, "cabb": 3, "cbaa": 4, "cbab": 5, "cbba": 6, "cbbb": 7, "cbca": 8, "cbcb": 9,}
    args.meta['si2ti'] = {
        0: [0, 1],
        1: [2, 3, 4],
    }
    args.meta['ti2ci'] = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
    }
    args.wtree = constants.get_wtree()
    batch_size = 32
    x_size = (53, 64, 52)

    # x = torch.randn((batch_size, *x_size), dtype=torch.float)
    svec = [np.random.randint(0, len(args.meta['s2i'])) for _ in range(batch_size)]
    tvec = [np.random.choice(args.meta['si2ti'][si]) for si in svec]
    cvec = [np.random.choice(args.meta['ti2ci'][ti]) for ti in tvec]

    svec = torch.tensor(svec, dtype=torch.long)
    tvec = torch.tensor(tvec, dtype=torch.long)
    cvec = torch.tensor(cvec, dtype=torch.long)

    for model_type, model_class in versions.items():
        print("Starting test on {}".format(model_type))
        gan = model_class(
            args,
            loadable_state_dict=None
        )

        # test generator
        z = torch.randn(batch_size, model.z_size).float()
        # test discriminator x->rf
        rf0, s, t, c = gan.discriminator(x)
        # test s prediction
        predict_s = torch.tensor([3])
        rf1, s, t, c = gan.discriminator(x, predict_s=predict_s)
        # test s, t
        rf, s, t, c = gan.discriminator(x, predict_s=predict_s, svec=svec)
        # test s, t, c
        rf, s, t, c = gan.discriminator(x, predict_s=predict_s, svec=svec, tvec=tvec)

        pdb.set_trace()
