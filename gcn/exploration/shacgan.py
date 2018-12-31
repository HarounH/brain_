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
import gcn.modules.generators as generators
import gcn.modules.discriminators as discriminators
import gcn.modules.gps as gps
import data.constants as constants

# X: N, 53, 64, 52
# Skeleton
# 5 downsampler: k4s2p1
# upsamplers are weird.

Z_SIZE = 128
CONTENT_CHANNELS = 16  # No task has more than 16 contrast, no study has more than 16 tasks? (expect brainomics->localizer)
DROPOUT_RATE = 0.5
LATENT_SIZE = Z_SIZE


class SHACGAN(nn.Module):
    LATENT_SIZE = Z_SIZE

    def __init__(self, args, gen_version='0', disc_version='0', gradient_penalty='dragan', loadable_state_dict=None):
        super(SHACGAN, self).__init__()
        self.args = args

        self.generator = generators.versions[gen_version](args, z_size=Z_SIZE, content_channels=CONTENT_CHANNELS, dropout_rate=DROPOUT_RATE)
        self.discriminator = discriminators.versions[disc_version](args, z_size=Z_SIZE, content_channels=CONTENT_CHANNELS, dropout_rate=DROPOUT_RATE)
        self.gradient_penalty = gps.versions[gradient_penalty](args.gp_lambda)

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)
        self.dataparallel = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def toggle_dataparallel(self):
        if self.dataparallel:
            self.dataparallel = False
            self.generator = self.generator.module
            self.discriminator = self.discriminator.module
            # self.gradient_penalty = self.gradient_penalty.module
        else:
            self.dataparallel = True
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
            # self.gradient_penalty = nn.DataParallel(self.gradient_penalty)
        return self


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

    gan = SHACGAN(
        args,
        gen_version='0',
        disc_version='0',
        gradient_penalty='dragan',
        loadable_state_dict=None
    )
    batch_size = 32
    x_size = constants.masked_nnz


    x = torch.randn((batch_size, constants.masked_nnz), dtype=torch.float)
    svec = [np.random.randint(0, len(args.meta['s2i'])) for _ in range(batch_size)]
    tvec = [np.random.choice(args.meta['si2ti'][si]) for si in svec]
    cvec = [np.random.choice(args.meta['ti2ci'][ti]) for ti in tvec]

    svec = torch.tensor(svec, dtype=torch.long)
    tvec = torch.tensor(tvec, dtype=torch.long)
    cvec = torch.tensor(cvec, dtype=torch.long)

    # pdb.set_trace()
    # test discriminator x->rf
    rf0, s, t, c = gan.discriminator(x)
    # pdb.set_trace()

    # test s prediction
    predict_s = torch.tensor([3])
    rf1, s, t, c = gan.discriminator(x, predict_s=predict_s)

    # test s, t
    rf, s, t, c = gan.discriminator(x, predict_s=predict_s, svec=svec)

    # test s, t, c
    rf, s, t, c = gan.discriminator(x, predict_s=predict_s, svec=svec, tvec=tvec)

    # test generator
    z = torch.randn(batch_size, Z_SIZE).float()
    x = gan.generator(z, svec, tvec, cvec)
    pdb.set_trace()
