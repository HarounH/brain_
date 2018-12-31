from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGL(nn.Module):
    # Fixed Graph Linear.
    # y = A * x * W
    # A (n_out * n_in sparse.FloatTensor): adjacency matrix - represents the graph that linear is faithful to
    # x: n_in * in_c FloatTensor
    # W: in_c * out_c FloatTensor
    # y: n_out * out_c FloatTensor
    def __init__(self, in_c, out_c, A, use_bias=True):
        super(FGL, self).__init__()
        n_out, n_in = A.shape
        self.in_c = in_c
        self.out_c = out_c
        self.n_in = n_in
        self.n_out = n_out

        indsA = A._indices()
        valsA = A._values()

        # row_degrees = []
        # _degrees = defaultdict(lambda: 0)
        # for i in inds[0, :]:
        #     _degrees[i] += 1
        # row_degrees.extend([(k, v) for k, v in _degrees.items()])
        # indsD = torch.tensor([[k, k] for (k, _) in row_degrees])
        # valsD = torch.tensor([v for (_, v) in row_degrees]).pow(-0.5)  # pow(-0.5) is safe since valsD > 0

        self.register_buffer('indsA', indsA)
        self.register_buffer('valsA', valsA)
        # self.register_buffer('A', A)
        self.A = A
        # self.register_buffer('indsD', indsD)
        # self.register_buffer('valsD', valsD)

        # Unfortunately, we don't have 2D sparse * 3D dense yet.
        # Hence, our best bet is to create a block sparse A, and multiply it with the
        # 2D version of input x (obtained by a simple x.view(-1, in_c)) or x.view(-1, out_c)
        blockA, _ = self.get_AD_(torch.zeros(32, self.n_in, self.in_c))
        # self.register_buffer('blockA', blockA)
        self.blockA = blockA
        # self.blockD = torch.sparse.FloatTensor(indsD.t(), valsD, size=self.blockA.shape)

        self.weight = nn.Parameter(torch.randn(in_c, out_c, dtype=torch.float) / n_out)
        if use_bias:  # It's a parameter in one case.
            self.bias = nn.Parameter(torch.randn(n_out, out_c, dtype=torch.float) / n_out)
        else:
            bias = torch.zeros(n_out, out_c).float()
            self.register_buffer('bias', bias)

    def get_AD_(self, x):
        if (not(hasattr(self, 'blockA'))) or (self.blockA.shape[1] != (x.shape[0] * self.n_in)):
            # Repeat A to match the size of x essentially.
            n_repeats = x.shape[0]
            blockA_inds = torch.cat([self.indsA + torch.tensor([[i * self.n_out], [i * self.n_in]]).to(self.indsA.device) for i in range(n_repeats)], dim=1)
            # blockD_inds = torch.cat([self.indsD + torch.tensor([[i], [i]]).to(self.indsD.device) for i in range(n_repeats)], dim=1)
            # Not ideal, but we can't use tensor.expand since its actual implementation doesn't favor us
            blockA_vals = self.valsA.repeat(n_repeats)
            # blockD_vals = self.valsD.repeat(n_repeats)
            blockA = torch.sparse.FloatTensor(blockA_inds, blockA_vals, size=(n_repeats * self.n_out, n_repeats * self.n_in))
            # self.blockD = torch.sparse.FloatTensor(blockD_inds, blockD_vals)
        else:
            blockA = self.blockA
        return blockA, None  # , self.blockD

    def forward(self, x):
        self.blockA, _ = self.get_AD_(x)
        if self.blockA.device != x.device:
            print("Called once")
            self.blockA = self.blockA.to(x.device)
        # return torch.stack([torch.spmm(self.A, t).mm(self.weight) for t in torch.unbind(x)]) + self.bias
        return (torch.spmm(self.blockA, x.view(-1, self.in_c)).mm(self.weight)).view(x.shape[0], self.n_out, self.out_c) + self.bias

if __name__ == '__main__':
    # Simple unit tests
    import pdb
    # Upsampling
    # n_out = 2, n_in=1...
    # out[i] is connected to floor(i / 2)
    def upsample_test():
        indsA = [[i, i//2] for i in range(2)]
        valsA = [1.0 for i in range(len(indsA))]
        indsA = torch.tensor(indsA).long()
        valsA = torch.tensor(valsA)
        A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(2, 1))
        fgl = FGL(1, 2, A)
        x = torch.randn(4, 1, 1)
        y = fgl(x)
        for i in range(x.shape[0]):
            assert (y[i] - A.to_dense().mm(x[i]).mm(fgl.weight) - fgl.bias).mean().abs().item() < 1e-6
        # pdb.set_trace()
        print("Completed upsample test")

    # Downsampling
    def downsample_test():
        indsA = [[i//2, i] for i in range(8)] + [[i // 2, 0] for i in range(8)]
        valsA = [1.0 for i in range(len(indsA))]
        indsA = torch.tensor(indsA).long()
        valsA = torch.tensor(valsA)
        A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(4, 8))
        fgl = FGL(1, 2, A)
        x = torch.randn(4, 8, 1)
        y = fgl(x)
        for i in range(x.shape[0]):
            assert (y[i] - A.to_dense().mm(x[i]).mm(fgl.weight) - fgl.bias).mean().abs().item() < 1e-6
        print("Completed downsample test")

    # Translation
    def translation_test():
        indsA = [[i, i] for i in range(8)] + [[i, 7 - i] for i in range(8)]
        valsA = [1.0 for i in range(len(indsA))]
        indsA = torch.tensor(indsA).long()
        valsA = torch.tensor(valsA)
        A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(8, 8))
        fgl = FGL(1, 2, A)
        x = torch.randn(4, 8, 1)
        y = fgl(x)
        for i in range(x.shape[0]):
            assert (y[i] - A.to_dense().mm(x[i]).mm(fgl.weight) - fgl.bias).mean().abs().item() < 1e-6
        print("Completed translation test")

    def cuda_test():
        indsA = [[i, i] for i in range(8)] + [[i, 7 - i] for i in range(8)]
        valsA = [1.0 for i in range(len(indsA))]
        indsA = torch.tensor(indsA).long().cuda()
        valsA = torch.tensor(valsA).cuda()
        A = torch.sparse.FloatTensor(indsA.t(), valsA, size=(8, 8))
        fgl = FGL(1, 2, A).cuda()
        x = torch.randn(4, 8, 1).cuda()
        y = fgl(x)
        for i in range(x.shape[0]):
            assert (y[i] - A.to_dense().mm(x[i]).mm(fgl.weight) - fgl.bias).mean().abs().item() < 1e-6
        print("Completed cuda test")

    upsample_test()
    downsample_test()
    translation_test()
    if torch.cuda.is_available():
        cuda_test()


    def profiling_test():
        N = 32
        nin = 32768
        cin = 64
        nout = 8192
        cout = 128
        # adj_list = [[i, 2 * i, 3 * i, 4 * i] for i in range(nout)]
        rows = torch.tensor([i // 4 for i in range(nin)]).long()
        cols = torch.tensor([i for i in range(nin)]).long()
        indsA = torch.stack([rows, cols]).long()
        valsA = torch.ones(nin).float()
        A = torch.sparse.FloatTensor(indsA, valsA, size=(nout, nin))
        fgl = FGL(cin, cout, A).cuda()

        x = torch.randn(N, nin, cin).cuda()
        y = fgl(x)
        x = torch.randn(N, nin, cin).cuda()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            y = fgl(x)
        print(prof)
    profiling_test()
