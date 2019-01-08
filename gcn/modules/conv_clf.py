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
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta

        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]
        if downsampled:
            self.z_size = z_size
            conv_net = []
            # 53, 64, 52
            conv_net.append(blocks.CC3D((53, 64, 52), 1, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample0
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation0
            conv_net.append(blocks.CC3D((26, 32, 26), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample1
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation1
            conv_net.append(blocks.CC3D((13, 16, 13), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample2
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation2
            # contrast
            conv_net.append(blocks.CC3D((6, 8, 6), z_size // 4, z_size // 2, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample3
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation3
            # task
            conv_net.append(blocks.CC3D((3, 4, 3), z_size // 2, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample4
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))  # self.activation4
            self.conv_net = nn.Sequential(*conv_net)
            self.ending_vol = 2
        else:
            self.z_size = z_size
            conv_net = []
            # 91, 109, 91
            conv_net.append(blocks.CC3D((91, 109, 91), 1, z_size // 32, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample0
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation0
            conv_net.append(blocks.CC3D((45, 54, 45), z_size // 32, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample1
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation1
            conv_net.append(blocks.CC3D((22, 27, 22), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample2
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation2
            # contrast
            conv_net.append(blocks.CC3D((11, 13, 11), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample3
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation3
            # task
            conv_net.append(blocks.CC3D((5, 6, 5), z_size // 4, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample4
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))
            self.conv_net = nn.Sequential(*conv_net)
            self.ending_vol = 2 * 3 * 2

        self.fc = nn.Sequential(
            nn.Linear(self.ending_vol * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        # for i in range(5):
        #     cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
        #     cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        cur_z = self.conv_net(cur_z)
        return self.fc(cur_z.view(N, -1))


class ConvClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]
        if downsampled:
            self.z_size = z_size
            # 53, 64, 52
            conv_net = []
            conv_net.append(nn.Conv3d(self.channel_sizes[0], self.channel_sizes[1], 4, stride=2, padding=1))  # self.downsample0
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation0
            conv_net.append(nn.Conv3d(self.channel_sizes[1], self.channel_sizes[2], 4, stride=2, padding=1))  # self.downsample1
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation1
            conv_net.append(nn.Conv3d(self.channel_sizes[2], self.channel_sizes[3], 4, stride=2, padding=1))  # self.downsample2
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation2
            # contrast
            conv_net.append(nn.Conv3d(self.channel_sizes[3], self.channel_sizes[4], 4, stride=2, padding=1))  # self.downsample3
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation3
            # task
            conv_net.append(nn.Conv3d(self.channel_sizes[4], self.channel_sizes[5], 4, stride=2, padding=1))  # self.downsample4
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))  # self.activation4
            self.conv_net = nn.Sequential(*conv_net)
            self.ending_vol = 2
        else:
            # 91, 109, 91
            self.z_size = z_size
            conv_net = []
            # 91, 109, 91
            conv_net.append(blocks.CC3D((91, 109, 91), 1, z_size // 32, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample0
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation0
            conv_net.append(blocks.CC3D((45, 54, 45), z_size // 32, z_size // 16, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample1
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation1
            conv_net.append(blocks.CC3D((22, 27, 22), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample2
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation2
            # contrast
            conv_net.append(blocks.CC3D((11, 13, 11), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample3
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2)))  # self.activation3
            # task
            conv_net.append(blocks.CC3D((5, 6, 5), z_size // 4, z_size, kernel=4, stride=2, padding=1, use_spectral_norm=True)) # self.downsample4
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))
            self.conv_net = nn.Sequential(*conv_net)
            self.ending_vol = 2 * 3 * 2

        self.fc = nn.Sequential(
            nn.Linear(self.ending_vol * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        # for i in range(5):
        #     cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
        #     cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        cur_z = self.conv_net(cur_z)
        return self.fc(cur_z.view(N, -1))
