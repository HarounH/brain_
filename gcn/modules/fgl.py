"""
Most of pytorch is N, C, * where C is channels.
In this file, we create FGL which takes N, C, H -> N, C', H'
"""


from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class FGL(nn.Module):
    # x: N, inc, inn
    # y: N, outc, outn
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
    def __init__(self, inc, inn, outc, outn, dict_adj_lists, reduction='sum', bias_type='', use_spectral_norm=False):
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
            str(k): maybe_spec_norm(FGL(inc, inn, outc, outn, v, bias_type=bias_type)) for k, v in dict_adj_lists.items()
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

    def test():
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
            [0, 0, 1, 1],
        ]).long()
        fgl = FGL(2, 4, 3, 5, adj_list)
        x = torch.randn(8, 4, 2)
        y = fgl(x)
        for i in range(x.shape[0]):
            assert (y[i] - A.float().mm(x[i]).mm(fgl.weight) - fgl.bias).mean().abs().item() < 1e-6
        # pdb.set_trace()
        print("Completed correctness test")

    # test()

    def profiling_test():
        N = 32
        nin = 32768
        cin = 64
        nout = 8192
        cout = 128
        adj_list = [[i, 2 * i, 3 * i, 4 * i] for i in range(nout)]
        fgl = FGL(cin, nin, cout, nout, adj_list).cuda()
        x = torch.randn(N, nin, cin).cuda()
        y = fgl(x)
        x = torch.randn(N, nin, cin).cuda()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            y = fgl(x)
        print(prof)
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     y.sum().backward()
        # print(prof)
    profiling_test()
