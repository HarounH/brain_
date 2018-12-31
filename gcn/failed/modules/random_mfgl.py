from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gcn.modules.fgl as fgl
import utils.utils as utils


class RandomMFGL(nn.Module):
    # Multiple Random perturbations of fixed Graph Linears
    def __init__(self, in_c, out_c, A, n=4, dA=0.05, use_bias=True):
        super(RandomMFGL, self).__init__()
        # assert(len(As) > 0)
        n_out, n_in = A.shape
        self.in_c = in_c
        self.out_c = out_c
        self.n_in = n_in
        self.n_out = n_out
        As = [A + utils.scsp2tsp(sp.rand(*(A.shape), dA).tocoo()) for i in range(n)]
        self.As = As
        self.nets = nn.ModuleList([fgl.FGL(in_c, out_c, A, use_bias) for A in As])

    def forward(self, x):
        cur = None
        for net in self.nets:
            if cur is not None:
                cur += net(x)
            else:
                cur = net(x)
        return cur / self.n
