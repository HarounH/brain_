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


class DiscriminatorHierarchical0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(DiscriminatorHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta

        # 53, 64, 52
        self.downsample0 = CC3D((53, 64, 52), 1, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample1 = CC3D((26, 32, 26), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample2 = CC3D((13, 16, 13), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        # contrast
        self.downsample3 = CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        # task
        self.downsample4 = CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))
        # study and real/fake
        # 1, 2, 1

        self.contrast_conv = nn.Sequential(
            CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.task_conv = nn.Sequential(
            CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.study_conv = nn.Sequential()

        self.contrast_fc = nn.Sequential(
            nn.Linear(3 * z_size * 1 * 2 * 1, len(meta['c2i'])),
        )
        self.task_fc = nn.Sequential(
            nn.Linear(2 * z_size * 1 * 2 * 1, len(meta['t2i'])),
        )
        self.study_fc = nn.Sequential(
            nn.Linear(1 * z_size * 1 * 2 * 1, len(meta['s2i'])),
        )
        self.rf_fc = nn.Sequential(
            nn.Linear(1 * z_size * 1 * 2 * 1, 1),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def construct_masks(self, meta):
        # TODO: Not used. Delete at some point
        si2ti_mask = np.zeros((len(meta['s2i']), len(meta['t2i'])), dtype=np.float32)
        for si in range(len(meta['s2i'])):
            for ti in meta['si2ti'][si]:
                si2ti_mask[si, ti] = 1.0
        si2ti_mask = torch.from_numpy(si2ti_mask)
        si2ti_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('si2ti_mask', si2ti_mask)

        ti2ci_mask = np.zeros((len(meta['t2i']), len(meta['c2i'])), dtype=np.float32)
        for ti in range(len(meta['t2i'])):
            for ci in meta['ti2ci'][ti]:
                ti2ci_mask[ti, ci] = 1.0
        ti2ci_mask = torch.from_numpy(ti2ci_mask)
        ti2ci_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('ti2ci_mask', ti2ci_mask)

    def forward(self, x, predict_s=None, svec=None, tvec=None):
        cur_z = x.unsqueeze(1)
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



class BigResolutionDiscriminatorHierarchical1(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(BigResolutionDiscriminatorHierarchical1, self).__init__()
        self.args = args
        meta = self.args.meta

        # 91, 109, 91
        self.downsample0 = CC3D((91, 109, 91), 1, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample1 = CC3D((45, 54, 45), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample2 = CC3D((22, 27, 22), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        # contrast
        self.downsample3 = CC3D((11, 13, 11), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        # task
        self.downsample4 = CC3D((5, 6, 5), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))
        # study and real/fake
        # 2, 3, 2

        self.contrast_conv = nn.Sequential(
            CC3D((11, 13, 11), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            CC3D((5, 6, 5), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.task_conv = nn.Sequential(
            CC3D((5, 6, 5), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.study_conv = nn.Sequential()

        self.contrast_fc = nn.Sequential(
            nn.Linear(3 * z_size * 2 * 3 * 2, len(meta['c2i'])),
        )
        self.task_fc = nn.Sequential(
            nn.Linear(2 * z_size * 2 * 3 * 2, len(meta['t2i'])),
        )
        self.study_fc = nn.Sequential(
            nn.Linear(1 * z_size * 2 * 3 * 2, len(meta['s2i'])),
        )
        self.rf_fc = nn.Sequential(
            nn.Linear(1 * z_size * 2 * 3 * 2, 1),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def construct_masks(self, meta):
        # TODO: Not used. Delete at some point
        si2ti_mask = np.zeros((len(meta['s2i']), len(meta['t2i'])), dtype=np.float32)
        for si in range(len(meta['s2i'])):
            for ti in meta['si2ti'][si]:
                si2ti_mask[si, ti] = 1.0
        si2ti_mask = torch.from_numpy(si2ti_mask)
        si2ti_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('si2ti_mask', si2ti_mask)

        ti2ci_mask = np.zeros((len(meta['t2i']), len(meta['c2i'])), dtype=np.float32)
        for ti in range(len(meta['t2i'])):
            for ci in meta['ti2ci'][ti]:
                ti2ci_mask[ti, ci] = 1.0
        ti2ci_mask = torch.from_numpy(ti2ci_mask)
        ti2ci_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('ti2ci_mask', ti2ci_mask)

    def forward(self, x, predict_s=None, svec=None, tvec=None):
        cur_z = x.unsqueeze(1)
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


class DiscriminatorHierarchical1(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(DiscriminatorHierarchical1, self).__init__()
        self.args = args
        meta = self.args.meta

        # 53, 64, 52
        self.smooth = nn.Sequential()  # spectral_norm(nn.Conv3d(1, 1, 3, stride=1, padding=1))
        self.downsample0 = CC3D((53, 64, 52), 1, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        # self.residual0 = ResidualBlock((26, 32, 26), z_size // 16, [3], use_spectral_norm=True)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))

        self.downsample1 = CC3D((26, 32, 26), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        # self.residual1 = ResidualBlock((13, 16, 13), z_size // 8, [3], use_spectral_norm=True)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))

        self.downsample2 = CC3D((13, 16, 13), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.residual2 = ResidualBlock((6, 8, 6), z_size // 4, [3], use_spectral_norm=True)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        # contrast
        self.downsample3 = CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.residual3 = ResidualBlock((3, 4, 3), z_size // 2, [3], use_spectral_norm=True)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        # task
        self.downsample4 = CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))
        # study and real/fake
        # 1, 2, 1

        self.contrast_conv = nn.Sequential(
            CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.task_conv = nn.Sequential(
            CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )  # 1, 2, 1
        self.study_conv = nn.Sequential()

        self.contrast_fc = nn.Sequential(
            nn.Linear(3 * z_size * 1 * 2 * 1, len(meta['c2i'])),
        )
        self.task_fc = nn.Sequential(
            nn.Linear(2 * z_size * 1 * 2 * 1, len(meta['t2i'])),
        )
        self.study_fc = nn.Sequential(
            nn.Linear(1 * z_size * 1 * 2 * 1, len(meta['s2i'])),
        )
        self.rf_fc = nn.Sequential(
            nn.Linear(1 * z_size * 1 * 2 * 1, 1),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def construct_masks(self, meta):
        # TODO: Not used. Delete at some point
        si2ti_mask = np.zeros((len(meta['s2i']), len(meta['t2i'])), dtype=np.float32)
        for si in range(len(meta['s2i'])):
            for ti in meta['si2ti'][si]:
                si2ti_mask[si, ti] = 1.0
        si2ti_mask = torch.from_numpy(si2ti_mask)
        si2ti_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('si2ti_mask', si2ti_mask)

        ti2ci_mask = np.zeros((len(meta['t2i']), len(meta['c2i'])), dtype=np.float32)
        for ti in range(len(meta['t2i'])):
            for ci in meta['ti2ci'][ti]:
                ti2ci_mask[ti, ci] = 1.0
        ti2ci_mask = torch.from_numpy(ti2ci_mask)
        ti2ci_mask.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('ti2ci_mask', ti2ci_mask)

    def forward(self, x, predict_s=None, svec=None, tvec=None):
        cur_z = self.smooth(x.unsqueeze(1))
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
    '1': DiscriminatorHierarchical1,
    'big0': BigResolutionDiscriminatorHierarchical1,
}
