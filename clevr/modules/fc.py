import numpy as np
import torch
from torch import nn


class Factored(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        super().__init__()
        # sizes = [args.meta['s'] ** 2, 4, args.meta['n_classes']]
        sizes = [args.meta['s'] ** 2, args.meta['n_classes']]
        net = []
        for i in range(len(sizes) - 1):
            net.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.net = nn.Sequential(*net)
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))
