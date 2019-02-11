from collections import defaultdict
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
import data.constants as constants
from conv.modules import blocks


class CoordConvClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta

        self.channel_sizes = [1, z_size // 32, z_size // 16, z_size // 8, z_size // 4, z_size]
        # self.channel_sizes = [1, 8, 16, 32, 32, 64]
        if downsampled:
            self.z_size = z_size
            conv_net = []
            # 53, 64, 52
            conv_net.append(blocks.CC3D((53, 64, 52), self.channel_sizes[0], self.channel_sizes[1], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample0
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation0
            conv_net.append(blocks.CC3D((26, 32, 26), self.channel_sizes[1], self.channel_sizes[2], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample1
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation1
            conv_net.append(blocks.CC3D((13, 16, 13), self.channel_sizes[2], self.channel_sizes[3], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample2
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation2
            conv_net.append(blocks.CC3D((6, 8, 6), self.channel_sizes[3], self.channel_sizes[4], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample3
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation3
            conv_net.append(blocks.CC3D((3, 4, 3), self.channel_sizes[4], self.channel_sizes[5], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample4
            if args.non_linear:
                conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))  # self.activation4

            self.ending_vol = 2
        else:
            self.z_size = z_size
            conv_net = []
            # 91, 109, 91
            conv_net.append(blocks.CC3D((91, 109, 91), self.channel_sizes[0], self.channel_sizes[1], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample0
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation0
            conv_net.append(blocks.CC3D((45, 54, 45), self.channel_sizes[1], self.channel_sizes[2], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample1
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation1
            conv_net.append(blocks.CC3D((22, 27, 22), self.channel_sizes[2], self.channel_sizes[3], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample2
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation2
            conv_net.append(blocks.CC3D((11, 13, 11), self.channel_sizes[3], self.channel_sizes[4], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample3
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation3
            # task
            conv_net.append(blocks.CC3D((5, 6, 5), self.channel_sizes[4], self.channel_sizes[5], kernel=4, stride=2, padding=1, use_weight_norm=True)) # self.downsample4
            if args.non_linear:
                conv_net.append(nn.Tanh())
            self.ending_vol = 2 * 3 * 2
        if args.dropout:
            conv_net.append(nn.Dropout())
        self.conv_net = nn.Sequential(*conv_net)

        fcs = []
        for i, study in enumerate(args.studies):
            si = args.meta['s2i'][study]
            assert(i == si)
            nclasses = len(args.meta['si2ci'][si])
            fcs.append(weight_norm(nn.Linear(self.ending_vol * self.channel_sizes[-1], nclasses)))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, study_vec, x):
        # x: N, 91, 109, 91
        N = x.shape[0]
        z = self.conv_net(x.unsqueeze(1))
        return self.fcs[study_vec[0].item()](z.view(N, -1))


class ConvClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [1, z_size // 32, z_size // 16, z_size // 8, z_size // 4, z_size]
        # self.channel_sizes = [1, 8, 16, 32, 32, 64]

        if downsampled:
            self.z_size = z_size
            # 53, 64, 52
            conv_net = []
            conv_net.append(nn.Conv3d(self.channel_sizes[0], self.channel_sizes[1], 4, stride=2, padding=1))  # self.downsample0
            # conv_net.append(nn.Tanh())  # self.activation0
            conv_net.append(nn.Conv3d(self.channel_sizes[1], self.channel_sizes[2], 4, stride=2, padding=1))  # self.downsample1
            # conv_net.append(nn.Tanh())  # self.activation1
            conv_net.append(nn.Conv3d(self.channel_sizes[2], self.channel_sizes[3], 4, stride=2, padding=1))  # self.downsample2
            # conv_net.append(nn.Tanh())  # self.activation2
            # contrast
            conv_net.append(nn.Conv3d(self.channel_sizes[3], self.channel_sizes[4], 4, stride=2, padding=1))  # self.downsample3
            # conv_net.append(nn.Tanh())  # self.activation3
            # task
            conv_net.append(nn.Conv3d(self.channel_sizes[4], self.channel_sizes[5], 4, stride=2, padding=1))  # self.downsample4
            # conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))  # self.activation4

            self.ending_vol = 2
        else:
            # 91, 109, 91
            self.z_size = z_size
            conv_net = []
            # 91, 109, 91
            conv_net.append(weight_norm(nn.Conv3d(self.channel_sizes[0], self.channel_sizes[1], 4, stride=2, padding=1))) # self.downsample0
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation0
            conv_net.append(weight_norm(nn.Conv3d(self.channel_sizes[1], self.channel_sizes[2], 4, stride=2, padding=1))) # self.downsample1
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation1
            conv_net.append(weight_norm(nn.Conv3d(self.channel_sizes[2], self.channel_sizes[3], 4, stride=2, padding=1))) # self.downsample2
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation2
            # contrast
            conv_net.append(weight_norm(nn.Conv3d(self.channel_sizes[3], self.channel_sizes[4], 4, stride=2, padding=1))) # self.downsample3
            if args.non_linear:
                conv_net.append(nn.Tanh())  # self.activation3
            # task
            conv_net.append(weight_norm(nn.Conv3d(self.channel_sizes[4], self.channel_sizes[5], 4, stride=2, padding=1))) # self.downsample4
            if args.non_linear:
                conv_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))
            self.ending_vol = 2 * 3 * 2

        conv_net.append(nn.Dropout())
        self.conv_net = nn.Sequential(*conv_net)

        fcs = []
        for i, study in enumerate(args.studies):
            si = args.meta['s2i'][study]
            assert(i == si)
            nclasses = len(args.meta['si2ci'][si])
            fcs.append(weight_norm(nn.Linear(self.ending_vol * self.channel_sizes[-1], nclasses)))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, study_vec, x):
        # x: N, 91, 109, 91
        N = x.shape[0]
        z = self.conv_net(x.unsqueeze(1))
        return self.fcs[study_vec[0].item()](z.view(N, -1))
