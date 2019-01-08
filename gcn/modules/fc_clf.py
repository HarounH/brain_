from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
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
            nn.Linear(int(in_features), len(args.meta['c2i']))
        )
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N, self.net[0].in_features
        return self.net(x)
