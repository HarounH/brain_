from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gcn.modules.fgl as fgl


class MFGL(nn.Module):
    # Multipe Fixed Graph Linears
    def __init__(self, in_c, out_c, As, use_bias=True):
        super(MFGL, self).__init__()
        # assert(len(As) > 0)
        n_out, n_in = As[0].shape
        self.in_c = in_c
        self.out_c = out_c
        self.n_in = n_in
        self.n_out = n_out
        self.As = As
        self.nets = nn.ModuleList([fgl.FGL(in_c, out_c, A, use_bias) for A in As])
        self.n = len(self.As)

    def forward(self, x):
        cur = None
        for net in self.nets:
            if cur is not None:
                cur += net(x)
            else:
                cur = net(x)
        return cur / self.n


if __name__ == '__main__':
    As = []
    indsA = [[i//2, i] for i in range(8)]
    valsA = [1.0 for i in range(len(indsA))]
    indsA = torch.tensor(indsA).long()
    valsA = torch.tensor(valsA)
    A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(4, 8))
    # fgl0 = fgl.FGL(1, 2, A)
    As.append(A)

    indsA = [[i // 2, 0] for i in range(8)]
    valsA = [1.0 for i in range(len(indsA))]
    indsA = torch.tensor(indsA).long()
    valsA = torch.tensor(valsA)
    A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(4, 8))
    # fgl1 = fgl.FGL(1, 2, A)
    As.append(A)
    mfgl = MFGL(1, 2, As)
    x = torch.randn(4, 8, 1)
    y = mfgl(x)
    for i in range(x.shape[0]):
        assert (y[i] - As[1].to_dense().mm(x[i]).mm(mfgl.nets[1].weight) - mfgl.nets[1].bias - As[0].to_dense().mm(x[i]).mm(mfgl.nets[0].weight) - mfgl.nets[0].bias).mean().abs().item() < 1e-6
    print("Completed MFGL test")
