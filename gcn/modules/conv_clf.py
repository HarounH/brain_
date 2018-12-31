from collections import defaultdict
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from gcn.modules import conv_blocks
import data.constants as constants
from conv.modules import blocks


class CoordConvClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super(ClassifierHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta

        self.z_size = z_size
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        # 53, 64, 52
        self.downsample0 = blocks.CC3D((53, 64, 52), 1, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample1 = blocks.CC3D((26, 32, 26), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample2 = blocks.CC3D((13, 16, 13), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        # contrast
        self.downsample3 = blocks.CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        # task
        self.downsample4 = blocks.CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))

        self.fc = nn.Sequential(
            nn.Linear(1 * 2 * 1 * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(5):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


class ConvClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super(ClassifierHierarchical0, self).__init__()
        self.args = args
        meta = self.args.meta

        self.z_size = z_size
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        # 53, 64, 52
        self.downsample0 = nn.Conv3d(self.channel_sizes[0], self.channel_sizes[1], 4, stride=2, padding=1)
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample1 = nn.Conv3d(self.channel_sizes[1], self.channel_sizes[2], 4, stride=2, padding=1)
        self.activation1 = nn.Sequential(nn.LeakyReLU(0.2))
        self.downsample2 = nn.Conv3d(self.channel_sizes[2], self.channel_sizes[3], 4, stride=2, padding=1)
        self.activation2 = nn.Sequential(nn.LeakyReLU(0.2))
        # contrast
        self.downsample3 = nn.Conv3d(self.channel_sizes[3], self.channel_sizes[4], 4, stride=2, padding=1)
        self.activation3 = nn.Sequential(nn.LeakyReLU(0.2))
        # task
        self.downsample4 = nn.Conv3d(self.channel_sizes[4], self.channel_sizes[5], 4, stride=2, padding=1)
        self.activation4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))

        self.fc = nn.Sequential(
            nn.Linear(1 * 2 * 1 * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(5):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))
