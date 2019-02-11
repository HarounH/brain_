import numpy as np
import torch
from torch import nn
from conv.modules import blocks


cs = [1, 2, 4, 8, 12, 16]


class ConvClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        self.args = args
        s = args.meta['s']
        net = []
        while True:
            layer_num = len(net)
            cin = cs[layer_num]
            cout = cs[layer_num + 1]
            net.append(nn.Conv2d(cin, cout, 2, stride=2, padding=0))
            s = s // 2
            if s == 1:
                break
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class CoordConvClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        self.args = args
        s = args.meta['s']
        net = []
        while True:
            layer_num = len(net)
            cin = cs[layer_num]
            cout = cs[layer_num + 1]
            net.append(blocks.CC2D((s, s), cin, cout, 2, stride=2, padding=0))
            s = s // 2
            if s == 1:
                break
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))
