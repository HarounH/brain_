import numpy as np
import torch
from torch import nn
from conv.modules import blocks
from gcn.modules import fgl


def quadrant_adjacency(s):
    quads = np.zeros(s).astype(np.int)
    quads[:s//2, :s//2] = 0
    quads[s//2:, :s//2] = 1
    quads[s//2:, s//2:] = 2
    quads[:s//2, s//2:] = 3
    quads = np.reshape(quads, (s * s, ))
    adj = [[] for _ in range(4)]
    for i in range(len(quads)):
        adj[quads[i]].append(i)
    return adj


class Classifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        super().__init__()
        self.args = args
        n_classes = args.meta['n_classes']
        s = args.meta['s']
        nout = 4
        cout = 4
        net = []
        adj = quadrant_adjacency(s)
        net.append(
            fgl.make_weight_normed_FGL(
                1,
                s * s,
                nout,
                cout,
                adj,
                "213",
                "sum",
                "tree",
            )
        )
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(Linear(nout * cout, n_classes))

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.unsqueeze(1).view(N, -1)).view(N, -1))
