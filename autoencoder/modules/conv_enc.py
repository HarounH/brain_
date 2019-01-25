import pdb
from math import floor
import numpy as np
# pytorch imports
import torch
import torch.nn as nn
from conv.modules import blocks
from torch.nn.utils import (
    spectral_norm,
    weight_norm,
)


class ConvEncoder0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [1, z_size // 32, z_size // 16, z_size // 8, z_size // 4, z_size]

        # 91, 109, 91
        self.z_size = z_size
        conv_net = []
        # 91, 109, 91
        conv_net.append(weight_norm(nn.Conv3d(1, z_size // 32, 4, stride=2, padding=1))) # self.downsample0
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation0
        # 45, 54, 45
        conv_net.append(weight_norm(nn.Conv3d(z_size // 32, z_size // 16, 4, stride=2, padding=1))) # self.downsample1
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation1
        # 22, 27, 22
        conv_net.append(weight_norm(nn.Conv3d(z_size // 16, z_size // 8, 4, stride=2, padding=1))) # self.downsample2
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation2
        # 11, 13, 11
        conv_net.append(weight_norm(nn.Conv3d(z_size // 8, z_size // 4, 4, stride=2, padding=1))) # self.downsample3
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation3
        # 5, 6, 5
        conv_net.append(weight_norm(nn.Conv3d(z_size // 4, z_size, 4, stride=2, padding=1))) # self.downsample4
        if args.non_linear:
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))
        self.conv_net = nn.Sequential(*conv_net)
        self.ending_vol = 2 * 3 * 2

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        return self.conv_net(cur_z).view(N, -1)


class CoordConvEncoder0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [1, z_size // 32, z_size // 16, z_size // 8, z_size // 4, z_size]

        # 91, 109, 91
        self.z_size = z_size
        conv_net = []
        # 91, 109, 91
        conv_net.append(blocks.CC3D((91, 109, 91), 1, z_size // 32, kernel=4, stride=2, padding=1, use_weight_norm=True))
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation0
        # 45, 54, 45
        conv_net.append(blocks.CC3D((45, 54, 45), z_size // 32, z_size // 16, kernel=4, stride=2, padding=1, use_weight_norm=True))
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation1
        # 22, 27, 22
        conv_net.append(blocks.CC3D((22, 27, 22), z_size // 16, z_size // 8, kernel=4, stride=2, padding=1, use_weight_norm=True))
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation2
        # 11, 13, 11
        conv_net.append(blocks.CC3D((11, 13, 11), z_size // 8, z_size // 4, kernel=4, stride=2, padding=1, use_weight_norm=True))
        if args.non_linear:
            conv_net.append(nn.Tanh())  # self.activation3
        # 5, 6, 5
        conv_net.append(blocks.CC3D((5, 6, 5), z_size // 4, z_size, kernel=4, stride=2, padding=1, use_weight_norm=True))
        if args.non_linear:
            conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))
        self.conv_net = nn.Sequential(*conv_net)
        self.ending_vol = 2 * 3 * 2

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        return self.conv_net(cur_z).view(N, -1)
