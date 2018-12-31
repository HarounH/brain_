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
import gcn.modules.fgl as fgl


class DiscriminatorHierarchical0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super(DiscriminatorHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.content_channels = content_channels
        self.z_size = z_size

        self.node_sizes = [constants.masked_nnz, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.nodes_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from 67615->32768...->128
        # we need to transpose each one and them reverse the list

        self.downsample0 = fgl.FGL(self.channel_sizes[0], self.node_sizes[0], self.channel_sizes[1], self.node_sizes[1], adj_list[0])
        self.downsample1 = fgl.FGL(self.channel_sizes[1], self.node_sizes[1], self.channel_sizes[2], self.node_sizes[2], adj_list[1])
        self.downsample2 = fgl.FGL(self.channel_sizes[2], self.node_sizes[2], self.channel_sizes[3], self.node_sizes[3], adj_list[2])
        self.downsample3 = fgl.FGL(self.channel_sizes[3], self.node_sizes[3], self.channel_sizes[4], self.node_sizes[4], adj_list[3])
        self.downsample4 = fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4])

        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2))

        self.contrast_downsample = nn.Sequential(
            fgl.FGL(self.channel_sizes[3], self.node_sizes[3], self.channel_sizes[4], self.node_sizes[4], adj_list[3]),
            nn.Sequential(nn.LeakyReLU(0.2)),
            fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4]),
            nn.Sequential(nn.LeakyReLU(0.2)),
        )
        self.task_downsample = nn.Sequential(
            fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4]),
            nn.Sequential(nn.LeakyReLU(0.2)),
        )
        self.study_downsample = nn.Sequential()
        self.contrast_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], 1),
            nn.Sigmoid(),
        )
        self.task_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], 1),
            nn.Sigmoid(),
        )
        self.study_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], 1),
            nn.Sigmoid(),
        )
        self.rf_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], 1),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x, predict_s=None, svec=None, tvec=None):
        # x: N, constants.masked_nnz
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        downsample = getattr(self, 'downsample{}'.format(0))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(0)):
            cur_z = getattr(self, 'residual{}'.format(0))(cur_z)
        if hasattr(self, 'activation{}'.format(0)):
            cur_z = getattr(self, 'activation{}'.format(0))(cur_z)
        downsample = getattr(self, 'downsample{}'.format(1))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(1)):
            cur_z = getattr(self, 'residual{}'.format(1))(cur_z)
        if hasattr(self, 'activation{}'.format(1)):
            cur_z = getattr(self, 'activation{}'.format(1))(cur_z)

        downsample = getattr(self, 'downsample{}'.format(2))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(2)):
            cur_z = getattr(self, 'residual{}'.format(2))(cur_z)
        if hasattr(self, 'activation{}'.format(2)):
            cur_z = getattr(self, 'activation{}'.format(2))(cur_z)

        c_z = cur_z
        downsample = getattr(self, 'downsample{}'.format(3))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(3)):
            cur_z = getattr(self, 'residual{}'.format(3))(cur_z)
        if hasattr(self, 'activation{}'.format(3)):
            cur_z = getattr(self, 'activation{}'.format(3))(cur_z)
        t_z = cur_z

        downsample = getattr(self, 'downsample{}'.format(4))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(4)):
            cur_z = getattr(self, 'residual{}'.format(4))(cur_z)
        if hasattr(self, 'activation{}'.format(4)):
            cur_z = getattr(self, 'activation{}'.format(4))(cur_z)
        rf_z = cur_z
        s_z = cur_z

        rf = self.rf_fc(rf_z.view(x.shape[0], -1))
        if predict_s is not None:
            s_z = self.study_conv(s_z.view(x.shape[0], -1))
            s = self.study_fc(s_z)

            if (svec is not None):
                t_z = self.task_conv(t_z).view(x.shape[0], -1)
                t = self.task_fc(torch.cat([s_z, t_z], dim=1).view(x.shape[0], -1))
                if (tvec is not None):
                    c_z = self.contrast_conv(c_z).view(x.shape[0], -1)
                    c = self.contrast_fc(torch.cat([s_z, t_z, c_z], dim=1).view(x.shape[0], -1))
                else:
                    c = torch.zeros(x.shape[0], device=self.args.device) - 1
            else:
                # Always predict study
                t = torch.zeros(x.shape[0], device=self.args.device) - 1
                c = torch.zeros(x.shape[0], device=self.args.device) - 1
        else:
            s = torch.zeros(x.shape[0], device=self.args.device) - 1
            t = torch.zeros(x.shape[0], device=self.args.device) - 1
            c = torch.zeros(x.shape[0], device=self.args.device) - 1

        return rf, s, t, c


versions = {
    '0': DiscriminatorHierarchical0,
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
    x_size = constants.masked_nnz

    x = torch.randn((batch_size, *x_size), dtype=torch.float)
    svec = [np.random.randint(0, len(args.meta['s2i'])) for _ in range(batch_size)]
    tvec = [np.random.choice(args.meta['si2ti'][si]) for si in svec]
    cvec = [np.random.choice(args.meta['ti2ci'][ti]) for ti in tvec]

    svec = torch.tensor(svec, dtype=torch.long)
    tvec = torch.tensor(tvec, dtype=torch.long)
    cvec = torch.tensor(cvec, dtype=torch.long)

    for model_type, model_class in versions.items():
        gan = model_class(
            args,
            loadable_state_dict=None
        )

        # test generator
        rf, s, t, c = gan.forward(x, svec, tvec, cvec)
        pdb.set_trace()
