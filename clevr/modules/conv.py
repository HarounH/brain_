import numpy as np
import torch
from torch import nn
from conv.modules import blocks


cs = [1, 2, 4, 6, 8, 10, 12, 14, 16]
small_cs = [1, 2, 2, 2, 2, 2, 2]



class HugeConv(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        use_cs = [1, 4, 32, 64, 128, 128, 128, 128, 128, 128, 128]
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net)
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(nn.Conv2d(cin, cout, 4, stride=2, padding=1))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class BigConv(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        use_cs = [1, 4, 32, 64, 128, 128, 128, 128, 128, 128, 128]
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net)
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(nn.Conv2d(cin, cout, 2, stride=2, padding=0))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class MaxBigConv(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        use_cs = [1, 2, 4, 16, 16, 16, 16, 16, 16, 16, 16]
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net) // 2
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(nn.Conv2d(cin, cout, 3, stride=1, padding=1))
            net.append(nn.MaxPool2d(2))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))



class BigCC(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        use_cs = [1, 4, 32, 64, 128, 128, 128, 128, 128, 128, 128]
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net)
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(blocks.CC2D((s, s), cin, cout, 2, stride=2, padding=0))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class MaxBigCC(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        use_cs = [1, 2, 4, 16, 16, 16, 16, 16, 16, 16, 16]
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net) // 2
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(blocks.CC2D((s, s), cin, cout, 3, stride=1, padding=1))
            net.append(nn.MaxPool2d(2))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class ConvClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, small=False):
        super().__init__()
        self.args = args
        s = args.meta['s']
        if s < 64:
            use_cs = small_cs
        else:
            use_cs = cs
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net)
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(nn.Conv2d(cin, cout, 2, stride=2, padding=0))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))


class CoordConvClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        super().__init__()
        self.args = args
        s = args.meta['s']
        if s < 64:
            use_cs = small_cs
        else:
            use_cs = cs
        net = []
        while True:
            if s == 1:
                break
            layer_num = len(net)
            cin = use_cs[layer_num]
            cout = use_cs[layer_num + 1]
            net.append(blocks.CC2D((s, s), cin, cout, 2, stride=2, padding=0))
            s = s // 2
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(cout, args.meta['n_classes']))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1)).view(N, -1))
