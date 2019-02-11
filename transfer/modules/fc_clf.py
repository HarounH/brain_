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
        self.sizes = [int(in_features), 256, 128]
        # self.sizes = [int(in_features), 128, len(args.meta['c2i'])]
        for i in range(1, len(self.sizes)):
            net.append(weight_norm(nn.Linear(self.sizes[i - 1], self.sizes[i])))
            if args.non_linear:
                net.append(nn.Tanh())

        if args.dropout:
            net.append(nn.Dropout())

        self.net = nn.Sequential(
            *(net)
        )

        fcs = []
        for i, study in enumerate(args.studies):
            si = args.meta['s2i'][study]
            assert(i == si)
            nclasses = len(args.meta['si2ci'][si])
            fcs.append(weight_norm(nn.Linear(self.sizes[-1], nclasses)))
        self.fcs = nn.ModuleList(fcs)

        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, study_vec, x):
        # x: N, self.net[0].in_features
        z = self.net(x)
        return self.fcs[study_vec[0].item()](z)
