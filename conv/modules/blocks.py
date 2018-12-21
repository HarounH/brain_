import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class CC3D(nn.Module):
    def __init__(self, volume, inc, outc, kernel, stride, padding, use_spectral_norm):
        super(CC3D, self).__init__()
        xgrid, ygrid, zgrid = np.meshgrid(np.linspace(-1, 1, volume[0]), np.linspace(-1, 1, volume[1]), np.linspace(-1, 1, volume[2]), indexing='ij')
        coords_ndarray = np.stack((xgrid, ygrid, zgrid), axis=0).astype('float32')
        coords = torch.from_numpy(coords_ndarray).unsqueeze(0)
        coords.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('coords', coords)
        if use_spectral_norm:
            self.conv = spectral_norm(nn.Conv3d(inc + 3, outc, kernel, stride=stride, padding=padding))
        else:
            self.conv = nn.Conv3d(inc + 3, outc, kernel, stride=stride, padding=padding)
        self.volume = volume
        self.inc = inc
        self.outc = outc

    def forward(self, x):
        return self.conv(torch.cat([x, self.coords.expand(x.shape[0], 3, *(self.volume))], dim=1))


class CCT3D(nn.Module):
    def __init__(self, volume, inc, outc, kernel, stride, padding, use_spectral_norm):
        super(CCT3D, self).__init__()
        xgrid, ygrid, zgrid = np.meshgrid(np.linspace(-1, 1, volume[0]), np.linspace(-1, 1, volume[1]), np.linspace(-1, 1, volume[2]), indexing='ij')
        coords_ndarray = np.stack((xgrid, ygrid, zgrid), axis=0).astype('float32')
        coords = torch.from_numpy(coords_ndarray).unsqueeze(0)
        coords.requires_grad = False  # 1, 3, *(target_shape)
        self.register_buffer('coords', coords)
        if use_spectral_norm:
            self.convt = spectral_norm(nn.ConvTranspose3d(inc + 3, outc, kernel, stride=stride, padding=padding))
        else:
            self.convt = nn.ConvTranspose3d(inc + 3, outc, kernel, stride=stride, padding=padding)

        self.volume = volume
        self.inc = inc
        self.outc = outc

    def forward(self, x):
        return self.convt(torch.cat([x, self.coords.expand(x.shape[0], 3, *(self.volume))], dim=1))


class ResidualBlock(nn.Module):  # Same for generator/discriminator
    def __init__(self, volume, c, ks, use_spectral_norm):
        super(ResidualBlock, self).__init__()
        self.c = c
        net = []
        for k in ks:
            net.extend([
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm3d(c),
                CC3D(volume, c, c, k, stride=1, padding=(k - 1) // 2, use_spectral_norm=use_spectral_norm),
            ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return x + self.net(x)
