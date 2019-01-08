"""
Most of pytorch is N, C, * where C is channels.
In this file, we create FGL which takes N, C, H -> N, C', H'

The correct classes to look at are:
FGL
RegionFGL
"""


from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import (
    spectral_norm,
    rnn,
)


class FGL(nn.Module):
    def __init__(self, inc, inn, outc, outn, adj_list, normalize=True, bias_type='nc', must_use_padded=False):
        """
        Does
            Equation: y = A (cdot(x, u)) v + b
            where:
            A is a sparse matrix of size (n_out, n_in) with learnable non-zero values (implemented as a dense mask and a dense parameter)
            cdot is hadamard
            u is a learnable weight matrix of size (n_in, c_in)
            v is a learnable weight matrix of size (c_in, c_out) ... I suspect this will help when we're trying to do the multi-region version with a summation
            b is a bias(edited)
        args
            inc (int): number of input channels
            inn (int): Number of input positions
            outc (int): Number of output channels
            outn (int): Number of output positions
            adj_list (list list int): outn lists containing lists of ints < inn
            bias_type (str):
                'none': no bias is used
                'c': a bias for each channel (same over output nodes)
                'nc': a bias for each channel and output node.
        """
        super(FGL, self).__init__()
        assert(bias_type in ['', 'c', 'nc'])
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        self.maxD = max(len(al) for al in adj_list)
        mask = torch.zeros(outn, self.maxD).float()  # Binary mask
        padded_adj_list = []
        for nidx, al in enumerate(adj_list):
            mask[nidx, :len(al)] = 1.0
            padded_adj_list.append(al + [0 for i in range(self.maxD - len(al))])

        lengths = mask.sum(dim=1, keepdim=True)
        if normalize:
            mask = mask / (1e-8 + lengths)

        A = torch.tensor(padded_adj_list).long()  # N, T

        with torch.no_grad():
            self.occupancy = (mask > 0).sum().item() / np.prod(mask.shape)
            if ((not(must_use_padded)) and (self.occupancy <= 0.25)):  # Arbitrary threshold
                print("FGL Optimization: Using packed. occupancy={}".format(self.occupancy))
                # Used as input to embedding matrix.
                lengths_int64 = lengths.squeeze().long()
                sorted_len, sorted_idx = lengths_int64.sort(0, descending=True)
                _, original_idx = sorted_idx.sort(0, descending=True)
                A_packed = rnn.pack_padded_sequence(A[sorted_idx, :], sorted_len, batch_first=True)

                self.register_buffer('A_packed_data', A_packed.data)
                self.A_packed_batch_sizes = A_packed.batch_sizes
                # self.register_buffer('A_packed_batch_sizes', A_packed.batch_sizes)
                self.register_buffer('sorted_len', sorted_len)
                self.register_buffer('sorted_idx', sorted_idx)
                self.register_buffer('original_idx', original_idx)  # Enable CUDA transfer

                # L = sum(lengths) ... for brains, = inn
                self.mask_weight_packed = nn.Parameter(0.2 * torch.randn((int(lengths.sum().item()), 1), dtype=torch.float))

                def get_almost_y(self, x):
                    # import pdb; pdb.set_trace()
                    N = x.shape[0]
                    x = x * self.weight
                    embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
                    embedding_output = F.embedding(self.A_packed_data, embedding_weight)  # L, (N * inc)
                    masked_embedding_output, _ = rnn.pad_packed_sequence(
                        rnn.PackedSequence(
                            data=self.mask_weight_packed * embedding_output,
                            batch_sizes=self.A_packed_batch_sizes,
                        ),
                        batch_first=True,
                    )
                    pooled_masked_embedding_output = masked_embedding_output.sum(dim=1)  # outn, (N * inc) but in wrong order.
                    gathered_pooled_masked_embedding_output = pooled_masked_embedding_output[self.original_idx]  # outn, (N * inc) in correct order.
                    pre_channel_transform = gathered_pooled_masked_embedding_output.view(self.outn, N, self.inc).contiguous().view(-1, self.inc)  # Flattened for linear
                    almost_y = self.channel_transform(pre_channel_transform).view(self.outn, N, self.outc)  # outn, N, outc
                    return almost_y

                # self.get_almost_y = self.packed_forward
            else:  # Padded
                # Used as input to embedding matrix.
                self.register_buffer('A', A)
                self.mask_weight = nn.Parameter(0.2 * torch.randn((outn, self.maxD, 1), dtype=torch.float))

                def get_almost_y(self, x):
                    N = x.shape[0]
                    x = (x * self.weight)
                    embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
                    embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N * inc)
                    masked_embedding_output = self.mask_weight * self.mask * embedding_output  # outn, maxD, N * inc
                    pooled_masked_embedding_output = masked_embedding_output.view(self.outn, self.maxD, N, self.inc).sum(dim=1).view(-1, self.inc)  # outn, N, inc
                    almost_y = self.channel_transform(pooled_masked_embedding_output).view(self.outn, N, self.outc)  # outn, N, outc
                    return almost_y
                # self.get_almost_y = self.padded_forward

        self.get_almost_y = get_almost_y
        mask = mask.unsqueeze(2)
        self.register_buffer('mask', mask)
        self.register_buffer('lengths', lengths)

        # parameters
        # self.mask_weight ... initialized based on occupancy
        self.weight = nn.Parameter(0.2 * torch.randn(inc, inn).float())  # u
        self.channel_transform = nn.Linear(inc, outc)  # v

        if bias_type == '':
            bias = torch.zeros(outc, 1).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, 1).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, outn).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

    def forward(self, x):
        # x: N, inc, inn
        # pdb.set_trace()
        almost_y = self.get_almost_y(self, x)
        y = almost_y.contiguous().permute(1, 2, 0).contiguous()
        return y + self.bias  # N, outc, outn

    def padded_forward(self, x):
        N = x.shape[0]
        x = (x * self.weight)
        embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
        embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N * inc)
        masked_embedding_output = self.mask_weight * self.mask * embedding_output  # outn, maxD, N * inc
        pooled_masked_embedding_output = masked_embedding_output.view(self.outn, self.maxD, N, self.inc).sum(dim=1).view(-1, self.inc)  # outn, N, inc
        almost_y = self.channel_transform(pooled_masked_embedding_output).view(self.outn, N, self.outc)  # outn, N, outc
        return almost_y

    def packed_forward(self, x):
        # import pdb; pdb.set_trace()
        N = x.shape[0]
        x = x * self.weight
        embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
        embedding_output = F.embedding(self.A_packed_data, embedding_weight)  # L, (N * inc)
        masked_embedding_output, _ = rnn.pad_packed_sequence(
            rnn.PackedSequence(
                data=self.mask_weight_packed * embedding_output,
                batch_sizes=self.A_packed_batch_sizes,
            ),
            batch_first=True,
        )
        pooled_masked_embedding_output = masked_embedding_output.sum(dim=1)  # outn, (N * inc) but in wrong order.
        gathered_pooled_masked_embedding_output = pooled_masked_embedding_output[self.original_idx]  # outn, (N * inc) in correct order.
        pre_channel_transform = gathered_pooled_masked_embedding_output.view(self.outn, N, self.inc).contiguous().view(-1, self.inc)  # Flattened for linear
        almost_y = self.channel_transform(pre_channel_transform).view(self.outn, N, self.outc)  # outn, N, outc
        return almost_y


class FGL_useless(nn.Module):
    # x: N, inc, inn
    # y: N, outc, outn
    def __init__(self, inc, inn, outc, outn, adj_list, normalize=True, bias_type=''):
        """
        args
            inc (int): number of input channels
            inn (int): Number of input positions
            outc (int): Number of output channels
            outn (int): Number of output positions
            adj_list (list list int): outn lists containing lists of ints < inn
            bias_type (str):
                'none': no bias is used
                'c': a bias for each channel (same over output nodes)
                'nc': a bias for each channel and output node.
        """
        super(FGL_useless, self).__init__()
        assert(bias_type in ['', 'c', 'nc'])
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        self.maxD = max(len(al) for al in adj_list)
        mask = torch.zeros(outn, self.maxD).float()  # Binary mask
        padded_adj_list = []
        for nidx, al in enumerate(adj_list):
            mask[nidx, :len(al)] = 1.0
            padded_adj_list.append(al + [0 for i in range(self.maxD - len(al))])
        if normalize:
            mask = mask / (1e-8 + mask.sum(dim=1, keepdim=True))
        mask = mask.unsqueeze(2)
        self.register_buffer('mask', mask)

        # Used as input to embedding matrix.
        A = torch.tensor(padded_adj_list).long()
        self.register_buffer('A', A)

        # parameters
        self.weight = nn.Parameter(0.2 * torch.randn(inc, outc).float())
        if bias_type == '':
            bias = torch.zeros(outc, 1).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, 1).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, outn).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

    def forward(self, x):
        # x: N, inc, inn
        N = x.shape[0]
        embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
        # embedding_weight = x.permute(1, 0, 2).contiguous().view(self.inn, -1)
        # embedding_weight = torch.reshape(x.permute(1, 0, 2), (self.inn, N * self.inc))
        embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N * inc)
        masked_embedding_output = self.mask * embedding_output  # outn, maxD, N * inc
        pooled_masked_embedding_output = masked_embedding_output.view(self.outn, self.maxD, N, self.inc).sum(dim=1)  # outn, N, inc
        almost_y = torch.bmm(pooled_masked_embedding_output, self.weight.unsqueeze(0).expand(self.outn, self.inc, self.outc))  # outn, N, outc
        y = almost_y.permute(1, 2, 0).contiguous()
        return y + self.bias  # N, outc, outn


class RegionFGL(nn.Module):
    def __init__(self, inc, inn, outc, outn, dict_adj_lists, normalize=True, reduction='sum', bias_type='', use_spectral_norm=False):
        """
        args
            inc (int): number of input channels
            inn (int): Number of input positions
            outc (int): Number of output channels
            outn (int): Number of output positions
            dict_adj_lists (dict of int-> list list int): a dictionary whose values are outn lists containing lists of ints < inn
            reduction (str): either '', 'sum', or 'mean'
                defines what pooling to use on the individual outputs from the various FGL modules
            bias_type (str):
                'none': no bias is used
                'c': a bias for each channel (same over output nodes)
                'nc': a bias for each channel and output node.
            use_spectral_norm (bool): Whether to use spectral_norm on each individual FGL
        """
        super(RegionFGL, self).__init__()
        assert(reduction in ['', 'sum', 'mean'])
        maybe_spec_norm = lambda mdl: spectral_norm(mdl) if use_spectral_norm else mdl
        self.fgls = nn.ModuleDict({
            str(k): maybe_spec_norm(FGL(inc, inn, outc, outn, v, normalize=normalize, bias_type=bias_type)) for k, v in dict_adj_lists.items()
        })
        self.nregions = len(dict_adj_lists)
        if reduction == '':
            self.reducer = lambda ydict: ydict
        elif reduction == 'sum':
            self.reducer = lambda ydict: sum(v for _, v in ydict.items())
        elif reduction == 'mean':
            self.reducer = lambda ydict: sum(v for _, v in ydict.items()) / self.nregions

    def forward(self, x, specific_region=None):
        if specific_region is None:
            return self.reducer({k: vmodel(x) for k, vmodel in self.fgls.items()})
        else:
            return {specific_region[0].item(): self.fgls[specific_region[0].item()](x)}


class FGL_node_first(nn.Module):
    # x: N, inn, inc
    # y: N, outn, outc
    def __init__(self, inc, inn, outc, outn, adj_list, bias_type=''):
        """
        args
            inc (int): number of input channels
            inn (int): Number of input positions
            outc (int): Number of output channels
            outn (int): Number of output positions
            adj_list (list list int): outn lists containing lists of ints < inn
            bias_type (str):
                'none': no bias is used
                'c': a bias for each channel (same over output nodes)
                'nc': a bias for each channel and output node.
        """
        super(FGL_node_first, self).__init__()
        assert(bias_type in ['', 'c', 'nc'])
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        self.maxD = max(len(al) for al in adj_list)
        mask = torch.zeros(outn, self.maxD).float()  # Binary mask
        padded_adj_list = []
        for nidx, al in enumerate(adj_list):
            mask[nidx, :len(al)] = 1.0
            padded_adj_list.append(al + [0 for i in range(self.maxD - len(al))])
        mask = mask.unsqueeze(2)
        self.register_buffer('mask', mask)

        # Used as input to embedding matrix.
        A = torch.tensor(padded_adj_list).long()
        self.register_buffer('A', A)

        # parameters
        self.weight = nn.Parameter(0.2 * torch.randn(inc, outc).float())
        if bias_type == '':
            bias = torch.zeros(outc).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(outc).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(outn, outc).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

    def forward(self, x):
        # x: N, inn, inc
        N = x.shape[0]
        embedding_weight = x.permute(1, 0, 2).contiguous().view(self.inn, -1)
        # embedding_weight = torch.reshape(x.permute(1, 0, 2), (self.inn, N * self.inc))
        embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N*inc)
        masked_embedding_output = self.mask * embedding_output
        pooled_masked_embedding_output = masked_embedding_output.view(self.outn, self.maxD, N, self.inc).sum(dim=1)  # outn, N, inc
        almost_y = torch.bmm(pooled_masked_embedding_output, self.weight.unsqueeze(0).expand(self.outn, self.inc, self.outc))  # outn, N, outc
        y = almost_y.permute(1, 0, 2).contiguous()
        return y + self.bias  # N, outn, outc


if __name__ == '__main__':
    import pdb

    def test_useless():
        adj_list = [
            [1],
            [2],
            [3],
            [0],
            [2, 3],
        ]
        A = torch.tensor([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0.5, 0.5],
        ])
        fgl = FGL_useless(2, 4, 3, 5, adj_list)
        x = torch.randn(8, 2, 4)
        y = fgl(x)
        for i in range(x.shape[0]):
            output = y[i]  # outc, outn
            correct = (A.float().mm(x[i].t()).mm(fgl.weight) - fgl.bias.t()).t()  # outc, outn
            assert (output - correct).abs().mean().item() < 1e-6
        print("Completed correctness test")

    def test_useless():
        adj_list = [
            [1],
            [2],
            [3],
            [0],
            [2, 3],
        ]
        A = torch.tensor([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0.5, 0.5],
        ])
        fgl = FGL_useless(2, 4, 3, 5, adj_list)
        x = torch.randn(8, 2, 4)
        y = fgl(x)
        for i in range(x.shape[0]):
            output = y[i]  # outc, outn
            correct = fgl.mask_weight * fgl.mask
            correct = (A.float().mm(x[i].t()).mm(fgl.weight) - fgl.bias.t()).t()  # outc, outn
            assert (output - correct).abs().mean().item() < 1e-6
        print("Completed correctness test")

    # test_useless()

    def warm_up():
        N = 32
        nin = 32768
        cin = 64
        nout = 2 * 8192
        cout = 128
        adj_list = [[i] for i in range(nout)]
        np.random.seed(1337)
        for i in range(nin):
            parent = np.random.randint(0, nout)
            if nin not in adj_list[parent]:  # Avoid duplicates.
                adj_list[parent].append(i)
        fgl = FGL(cin, nin, cout, nout, adj_list).cuda()
        print("Occupancy = {}".format(fgl.occupancy))
        x = torch.randn(N, cin, nin).cuda()
        y = fgl(x)
        x = torch.randn(N, cin, nin).cuda()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = fgl(x)
        # print(prof)

    def profiling_test(must_use_padded):
        N = 32
        nin = 32768
        cin = 64
        nout = 2 * 8192
        cout = 128
        adj_list = [[i] for i in range(nout)]
        np.random.seed(1337)
        for i in range(nin):
            parent = np.random.randint(0, nout)
            if nin not in adj_list[parent]:  # Avoid duplicates.
                adj_list[parent].append(i)
        fgl = FGL(cin, nin, cout, nout, adj_list, must_use_padded=must_use_padded).cuda()
        print("Occupancy = {}".format(fgl.occupancy))
        x = torch.randn(N, cin, nin).cuda()
        y = fgl(x)
        x = torch.randn(N, cin, nin).cuda()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            y = fgl(x)
        print(prof)
        # pdb.set_trace()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     y.sum().backward()
        # print(prof)
    import time

    warm_up()
    warm_up()
    warm_up()
    warm_up()

    print("Using padded:")
    start = time.time()
    profiling_test(True)
    padded_time = time.time() - start

    print("Using packed:")
    start = time.time()
    profiling_test(False)
    packed_time = time.time() - start

    print("Packed={} v/s Padded={}".format(packed_time, padded_time))
