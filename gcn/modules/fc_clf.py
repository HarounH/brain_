from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Classifier(nn.Module):
    def __init__(self, sizes, activation_layer_maker=lambda i: nn.Tanh):
        """
            args
            sizes: list int
            activation_layer_maker: int (op #) -> (fn: None -> nn.Module)
        """
        super(Classifier, self).__init__()
        net = []
        for i in range(len(sizes) - 1):
            net.append(nn.Linear(sizes[i], sizes[i + 1]))
            net.append(activation_layer_maker(i)())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # x: N, self.net[0].in_features
        return self.net(x)
