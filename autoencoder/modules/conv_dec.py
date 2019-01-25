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
from data.constants import original_brain_mask_tensor


class ConvDecoder0(nn.Module):  # Only for MNI152 really.
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [z_size, z_size // 4, z_size // 8, z_size // 16, z_size // 32, 1]
        self.z_size = z_size
        self.emb_shape = (2, 3, 2)
        self.emb_vol = np.prod(self.emb_shape)
        convt_net = []
        # 2, 3, 2
        convt_net.append(weight_norm(nn.ConvTranspose3d(self.channel_sizes[0], self.channel_sizes[1], (3, 4, 3), stride=2, padding=(0, 1, 0))))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation0
        # 5, 6, 5
        convt_net.append(weight_norm(nn.ConvTranspose3d(self.channel_sizes[1], self.channel_sizes[2], (3, 3, 3), stride=2, padding=(0, 0, 0))))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation1
        # 11, 13, 11
        convt_net.append(weight_norm(nn.ConvTranspose3d(self.channel_sizes[2], self.channel_sizes[3], (4, 3, 4), stride=2, padding=(1, 0, 1))))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation2
        # 22, 27, 22
        convt_net.append(weight_norm(nn.ConvTranspose3d(self.channel_sizes[3], self.channel_sizes[4], (3, 4, 3), stride=2, padding=(0, 1, 0))))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation3
        # 45, 54, 45
        convt_net.append(weight_norm(nn.ConvTranspose3d(self.channel_sizes[4], self.channel_sizes[5], (3, 3, 3), stride=2, padding=(0, 0, 0))))
        # 91, 109, 91
        if args.non_linear:
            convt_net.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(dropout_rate)))

        self.convt_net = nn.Sequential(*convt_net)

    def forward(self, z):
        # import pdb; pdb.set_trace()
        # x: N, (z_size * ending_vol)
        N = z.shape[0]
        cur_z = z.view(N, self.z_size, *(self.emb_shape))
        return self.convt_net(cur_z).view(N, 91, 109, 91)


class CoordConvDecoder0(nn.Module):  # Only for MNI152 really.
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        self.channel_sizes = [z_size, z_size // 4, z_size // 8, z_size // 16, z_size // 32, 1]

        self.z_size = z_size
        self.emb_shape = (2, 3, 2)
        self.emb_vol = np.prod(self.emb_shape)
        convt_net = []
        # 2, 3, 2
        convt_net.append(blocks.CCT3D((2, 3, 2), self.channel_sizes[0], self.channel_sizes[1], (3, 4, 3), stride=2, padding=(0, 1, 0), use_weight_norm=True))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation0
        # 5, 6, 5
        convt_net.append(blocks.CCT3D((5, 6, 5), self.channel_sizes[1], self.channel_sizes[2], (3, 3, 3), stride=2, padding=(0, 0, 0), use_weight_norm=True))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation1
        # 11, 13, 11
        convt_net.append(blocks.CCT3D((11, 13, 11), self.channel_sizes[2], self.channel_sizes[3], (4, 3, 4), stride=2, padding=(1, 0, 1), use_weight_norm=True))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation2
        # 22, 27, 22
        convt_net.append(blocks.CCT3D((22, 27, 22), self.channel_sizes[3], self.channel_sizes[4], (3, 4, 3), stride=2, padding=(0, 1, 0), use_weight_norm=True))
        if args.non_linear:
            convt_net.append(nn.Tanh())  # self.activation3
        # 45, 54, 45
        convt_net.append(blocks.CCT3D((45, 54, 45), self.channel_sizes[4], self.channel_sizes[5], (3, 3, 3), stride=2, padding=(0, 0, 0), use_weight_norm=True))
        # 91, 109, 91
        self.convt_net = nn.Sequential(*convt_net)

        mask = original_brain_mask_tensor
        self.register_buffer('mask', mask)

    def forward(self, z):
        # import pdb; pdb.set_trace()
        # x: N, (z_size * ending_vol)
        N = z.shape[0]
        cur_z = z.view(N, self.z_size, *(self.emb_shape))
        return self.convt_net(cur_z).view(N, 91, 109, 91) * self.mask
