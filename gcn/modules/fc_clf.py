from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import (
    weight_norm,
    spectral_norm,
)
import data.constants as constants


class Classifier(nn.Module):
    def __init__(self, args, activation_layer_maker=lambda i: nn.Tanh, loadable_state_dict=None, downsampled=False):
        """
            args
            sizes: list int
            activation_layer_maker: int (op #) -> (fn: None -> nn.Module)
        """
        super().__init__()
        self.args = args
        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        self.net = nn.Sequential(
            weight_norm(nn.Linear(int(in_features), len(args.meta['c2i'])))
        )
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N, self.net[0].in_features
        return self.net(x)


class Linear(nn.Module):
    def __init__(self, args, activation_layer_maker=lambda i: nn.Tanh, loadable_state_dict=None, downsampled=False):
        """
            args
            sizes: list int
            activation_layer_maker: int (op #) -> (fn: None -> nn.Module)
        """
        super().__init__()
        self.args = args
        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        net = []
        self.sizes = [int(in_features), 512, 256, 128, len(args.meta['c2i'])]
        # self.sizes = [int(in_features), 128, len(args.meta['c2i'])]
        for i in range(1, len(self.sizes)):
            net.append(weight_norm(nn.Linear(self.sizes[i - 1], self.sizes[i])))
            if args.non_linear:
                net.append(nn.Tanh())
        self.net = nn.Sequential(
            *(net)
        )
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N, self.net[0].in_features
        return self.net(x)


class DimensionReduced(nn.Module):
    def __init__(self, args, activation_layer_maker=lambda i: nn.Tanh, loadable_state_dict=None, downsampled=False):
        """
            args
            sizes: list int
            activation_layer_maker: int (op #) -> (fn: None -> nn.Module)
        """
        super().__init__()
        self.args = args
        self.wtree = args.wtree
        self.in_features = in_features = 16000  # Tunable
        labelling = self.wtree.cut(in_features)
        self.lengths = torch.tensor(np.unique(labelling, return_counts=True)[1]).float()

        self.labelling = torch.tensor(labelling).long().unsqueeze(0)
        self.net = nn.Sequential(
            weight_norm(nn.Linear(int(in_features), len(args.meta['c2i'])))
        )
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)
        self.reduced_x = None


    def forward(self, x):
        # x: N, self.net[0].in_features
        N = x.shape[0]
        if self.reduced_x is None or N != self.reduced_x.shape[0]:
            self.reduced_x = torch.zeros(x.shape[0], self.in_features, device=x.device)
        else:
            self.reduced_x.data.zero_()
        self.reduced_x.scatter_add_(1, self.labelling.expand(x.shape).to(x.device), x)
        self.reduced_x = self.reduced_x / self.lengths.to(self.reduced_x.device)
        return self.net(self.reduced_x)
