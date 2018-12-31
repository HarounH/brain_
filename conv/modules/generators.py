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
from .blocks import (
    CC3D,
    CCT3D,
    ResidualBlock,
)


class GeneratorHierarchical0(nn.Module):
    def __init__(self, args, mask=None, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(GeneratorHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta
        self.z_size = z_size
        self.content_channels = content_channels
        # Dont need any constraints here - the constraints are imposed by original choice of s, t, c during training.
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            self.mask = mask
        else:
            self.mask = None
        self.zfc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
        )

        self.study_embedding = weight_norm(nn.Embedding(len(meta['s2i']), content_channels))
        self.task_embedding = weight_norm(nn.Embedding(len(meta['t2i']), content_channels))
        self.contrast_embedding = weight_norm(nn.Embedding(len(meta['c2i']), content_channels))

        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(2 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
        ])

        # (1, 2, 1) - k343s2p010 - (3, 4, 3) - k4s2p1 - (6, 8, 6) - k343s2p010 - (13, 16, 13) - k4s2p1 - (26, 32, 26) - k344s2p011 - 53, 64, 52
        # Now, to convolve.
        self.upsample0 = CCT3D((1, 2, 1), z_size + content_channels, z_size // 2, (3, 4, 3), stride=2, padding=(0, 1, 0), use_spectral_norm=False)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 2))

        self.upsample1 = CCT3D((3, 4, 3), z_size // 2 + content_channels, z_size // 4, (4, 4, 4), stride=2, padding=(1, 1, 1), use_spectral_norm=False)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 4))
        self.residual1 = ResidualBlock((6, 8, 6), z_size // 4, [3], use_spectral_norm=False)

        self.upsample2 = CCT3D((6, 8, 6), z_size // 4 + content_channels, z_size // 8, (3, 4, 3), stride=2, padding=(0, 1, 0), use_spectral_norm=False)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 8))

        self.upsample3 = CCT3D((13, 16, 13), z_size // 8 + content_channels, z_size // 16, (4, 4, 4), stride=2, padding=(1, 1, 1), use_spectral_norm=False)
        self.residual3 = ResidualBlock((26, 32, 26), z_size // 16, [3], use_spectral_norm=False)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 16))

        self.upsample4 = CCT3D((26, 32, 26), z_size // 16 + content_channels, 1, (3, 4, 4), stride=2, padding=(0, 1, 1), use_spectral_norm=False)
        self.activation4 = nn.Sequential(nn.Tanh())

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, z, studies, tasks, contrasts):
        # z: N, z_size

        se = self.study_embedding(studies)
        te = self.task_embedding(tasks)
        ce = self.contrast_embedding(contrasts)
        contents = [
            se,
            torch.cat([se, te], dim=1),  # se + te,
            torch.cat([se, te, ce], dim=1),  # se + te + ce
        ]
        contents.append(contents[-1])
        contents.append(contents[-1])
        contents = [self.fcs[i](content) for i, content in enumerate(contents)]

        # Now, add the contents to z as it goes through the CNN
        cur_z = self.zfc(z).unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(z.shape[0], z.shape[1], 1, 2, 1)
        for i in range(0, 5):
            upsample = getattr(self, 'upsample{}'.format(i))
            content = contents[i].unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(z.shape[0], self.content_channels, cur_z.shape[2], cur_z.shape[3], cur_z.shape[4])
            cur_z = upsample(torch.cat([cur_z, content], dim=1))
            if hasattr(self, 'residual{}'.format(i)):
                cur_z = getattr(self, 'residual{}'.format(i))(cur_z)
            if hasattr(self, 'activation{}'.format(i)):
                cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        retval = cur_z[:, 0, ...]
        if self.mask is not None:
            return retval * self.mask.expand(retval.shape).to(retval.device)
        else:
            return retval


class GeneratorHierarchical1(nn.Module):
    def __init__(self, args, mask=None, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(GeneratorHierarchical1, self).__init__()
        self.args = args
        meta = self.args.meta
        self.z_size = z_size
        self.content_channels = content_channels
        # Dont need any constraints here - the constraints are imposed by original choice of s, t, c during training.
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(0)
            self.mask = mask
        else:
            self.mask = None
        self.zfc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
        )

        self.study_embedding = weight_norm(nn.Embedding(len(meta['s2i']), content_channels))
        self.task_embedding = weight_norm(nn.Embedding(len(meta['t2i']), content_channels))
        self.contrast_embedding = weight_norm(nn.Embedding(len(meta['c2i']), content_channels))

        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(2 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
        ])

        # (1, 2, 1) - k343s2p010 - (3, 4, 3) - k4s2p1 - (6, 8, 6) - k343s2p010 - (13, 16, 13) - k4s2p1 - (26, 32, 26) - k344s2p011 - 53, 64, 52
        # Now, to convolve.
        self.upsample0 = CCT3D((1, 2, 1), z_size + content_channels, z_size // 2, (3, 4, 3), stride=2, padding=(0, 1, 0), use_spectral_norm=False)
        # self.residual0 = ResidualBlock((3, 4, 3), z_size // 2, [3], use_spectral_norm=False)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 2))

        self.upsample1 = CCT3D((3, 4, 3), z_size // 2 + content_channels, z_size // 4, (4, 4, 4), stride=2, padding=(1, 1, 1), use_spectral_norm=False)
        # self.residual1 = ResidualBlock((6, 8, 6), z_size // 4, [3], use_spectral_norm=False)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 4))

        self.upsample2 = CCT3D((6, 8, 6), z_size // 4 + content_channels, z_size // 8, (3, 4, 3), stride=2, padding=(0, 1, 0), use_spectral_norm=False)
        # self.residual2 = ResidualBlock((13, 16, 13), z_size // 8, [3], use_spectral_norm=False)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 8))

        self.upsample3 = CCT3D((13, 16, 13), z_size // 8 + content_channels, z_size // 16, (4, 4, 4), stride=2, padding=(1, 1, 1), use_spectral_norm=False)
        self.residual3 = ResidualBlock((26, 32, 26), z_size // 16, [3], use_spectral_norm=False)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2), nn.BatchNorm3d(z_size // 16))

        self.upsample4 = CCT3D((26, 32, 26), z_size // 16 + content_channels, 1, (3, 4, 4), stride=2, padding=(0, 1, 1), use_spectral_norm=False)
        self.residual4 = nn.Conv3d(1, 1, 3, stride=1, padding=1)  # Not a residual...
        self.activation4 = nn.Sequential(nn.Tanh())

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, z, studies, tasks, contrasts):
        # z: N, z_size

        se = self.study_embedding(studies)
        te = self.task_embedding(tasks)
        ce = self.contrast_embedding(contrasts)
        contents = [
            se,
            torch.cat([se, te], dim=1),  # se + te,
            torch.cat([se, te, ce], dim=1),  # se + te + ce
        ]
        contents.append(contents[-1])
        contents.append(contents[-1])
        contents = [self.fcs[i](content) for i, content in enumerate(contents)]

        # Now, add the contents to z as it goes through the CNN
        cur_z = self.zfc(z).unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(z.shape[0], z.shape[1], 1, 2, 1)
        for i in range(0, 5):
            upsample = getattr(self, 'upsample{}'.format(i))
            content = contents[i].unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(z.shape[0], self.content_channels, cur_z.shape[2], cur_z.shape[3], cur_z.shape[4])
            cur_z = upsample(torch.cat([cur_z, content], dim=1))
            if hasattr(self, 'residual{}'.format(i)):
                cur_z = getattr(self, 'residual{}'.format(i))(cur_z)
            if hasattr(self, 'activation{}'.format(i)):
                cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        retval = cur_z[:, 0, ...]
        if self.mask is not None:
            return retval * self.mask.expand(retval.shape).to(retval.device)
        else:
            return retval


versions = {
    '0': GeneratorHierarchical0,
    '1': GeneratorHierarchical1,
}
