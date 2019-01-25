"""
This file is deprecated because it was refactored. Refer to gcn.modules.fgl
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
    weight_norm,
    rnn,
)
import utils.utils as utils


DENSITY_THRESHOLD = 0.25  # If sparse adjacency matrix, A, is less occupied than this, then packed tensors are used instead of padded for embedding (which is the bottleneck)


class FGL(nn.Module):
    def __init__(self, inc, inn, outc, outn, adj_list, A_options='none_norm', bias_type='nc', must_use_padded=False, tree_optimization=False):
        """
        Does
            Equation: y = (A (cdot(x.T, u)) v + b).T
            where:
            A is a sparse matrix of size (n_out, n_in) with (potentially) learnable non-zero values (implemented as a dense mask and a dense parameter)
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
            A_options (str):
                root:
                    'none': not learnable
                    'sparse': only the values are learnable
                suffix:
                    To normalize A by degree, suffix with "_norm"
                prefix:
                    N/A
            tree_optimization (bool): If its a tree then each input node will
                belong to only one output node. This allows us to use scatter_add_
        """
        super(FGL, self).__init__()
        assert(bias_type in ['', 'c', 'nc'])
        normalize = A_options.endswith("_norm")

        if normalize:
            A_options = A_options[:-5]
        # if A_options.startswith("max_"):
        #     use_max_pool = True
        #     A_options = A_options[4:]
        #     raise NotImplementedError()
        # else:
        #     use_max_pool = False

        use_mask_weight = A_options in ["sparse"]
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        self.maxD = max(len(al) for al in adj_list)

        # parameters
        # self.mask_weight ... initialized based on density
        self.weight = nn.Parameter((np.sqrt(2 / inn)) * (-1 + 2 * torch.rand(inc, inn)).float())  # u
        if tree_optimization:
            self.channel_transform = nn.Parameter((-1 + 2 * torch.rand(inc, outc).float()) * np.sqrt(2 / inc))
        else:
            self.channel_transform = weight_norm(nn.Linear(inc, outc, bias=False))  # v

        if bias_type == '':
            bias = torch.zeros(1).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(1, outc, 1).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(1, outc, outn).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

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
            self.density = (mask > 0).sum().item() / np.prod(mask.shape)
            if tree_optimization:
                print("Using tree optimization")
                if use_mask_weight:
                    raise NotImplementedError("Can't use learnable mask weight with tree optimization.")
                self.pre_channel_transform = None
                indices = torch.tensor(utils.transpose_adj_list(self.outn, self.inn, adj_list)).long().t().unsqueeze(0)
                self.register_buffer('indices', indices)
                self.normalize = normalize

                def forward_(self, x):
                    N = x.shape[0]
                    x = x * self.weight  # Unavoidable cost.
                    if self.pre_channel_transform is None or self.pre_channel_transform.shape[0] != N:
                        self.pre_channel_transform = torch.zeros(N, self.inc, self.outn)
                    else:
                        self.pre_channel_transform.data.zero_()
                    self.pre_channel_transform = self.pre_channel_transform.to(x.device)
                    self.pre_channel_transform.scatter_add_(2, self.indices.to(x.device).expand(x.shape), x)  # N, inc, outn
                    if self.normalize:
                        self.pre_channel_transform = self.pre_channel_transform / self.lengths[:, 0]
                    y = torch.einsum("...ij,ik->...kj", self.pre_channel_transform, self.channel_transform)
                    import pdb; pdb.set_trace()
                    return y + self.bias

            elif ((not(must_use_padded)) and (self.density <= DENSITY_THRESHOLD)):  # Arbitrary threshold
                print("FGL Optimization: Using packed. density={}".format(self.density))
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
                if use_mask_weight:
                    self.mask_weight_packed = nn.Parameter(torch.randn((int(lengths.sum().item()), 1), dtype=torch.float))

                def forward_(self, x):
                    # x: N, inc, inn
                    # import pdb; pdb.set_trace()
                    N = x.shape[0]
                    x = x * self.weight  # N, inc, inn
                    embedding_weight = x.view(-1, self.inn).t()  # inn, (N * inc)
                    embedding_output = F.embedding(self.A_packed_data, embedding_weight)  # L, (N * inc)
                    masked_embedding_output, _ = rnn.pad_packed_sequence(
                        rnn.PackedSequence(
                            data=self.mask_weight_packed * embedding_output if use_mask_weight else embedding_output,
                            batch_sizes=self.A_packed_batch_sizes,
                        ),
                        batch_first=True,
                    )  # outn, D, (N * inc)
                    pooled_masked_embedding_output = masked_embedding_output.sum(dim=1)  # outn, (N * inc) but in wrong order.
                    gathered_pooled_masked_embedding_output = pooled_masked_embedding_output[self.original_idx]  # outn, (N * inc) in correct order.
                    pre_channel_transform = gathered_pooled_masked_embedding_output.view(-1, self.inc)  # .view(self.outn, N, self.inc).contiguous()  # Flattened for linear
                    almost_y = self.channel_transform(pre_channel_transform).view(self.outn, N, self.outc)  # outn, N, outc
                    y = almost_y.contiguous().permute(1, 2, 0).contiguous()
                    return y + self.bias

                # self.get_almost_y = self.packed_forward
            else:  # Padded
                # Used as input to embedding matrix.
                self.register_buffer('A', A)
                if use_mask_weight:
                    self.mask_weight = nn.Parameter(torch.randn((outn, self.maxD, 1), dtype=torch.float))

                def forward_(self, x):
                    N = x.shape[0]
                    x = (x * self.weight)
                    embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
                    embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N * inc)
                    masked_embedding_output = self.mask_weight * self.mask * embedding_output if use_mask_weight else self.mask * embedding_output  # outn, maxD, N * inc
                    pooled_masked_embedding_output = masked_embedding_output.sum(dim=1).view(-1, self.inc)  # .view(self.outn, self.maxD, N, self.inc).sum(dim=1).view(-1, self.inc)  # outn, N, inc
                    almost_y = self.channel_transform(pooled_masked_embedding_output).view(self.outn, N, self.outc)  # outn, N, outc
                    y = almost_y.contiguous().permute(1, 2, 0).contiguous()
                    return y + self.bias

                # self.get_almost_y = self.padded_forward

        self.forward_ = forward_
        mask = mask.unsqueeze(2)
        self.register_buffer('mask', mask)
        self.register_buffer('lengths', lengths)

    def forward(self, x):
        # x: N, inc, inn
        return self.forward_(self, x)


class FGL_v2(nn.Module):
    def __init__(self, inc, inn, outc, outn, adj_list, A_options='none_norm', bias_type='nc', must_use_padded=False, *args, **kwargs):
        """
        Does
            Equation:
                y = hadamard(Ax, u)v + b
            where:
            A is a sparse matrix of size (n_out, n_in) with (potentially) learnable non-zero values (implemented as a dense mask and a dense parameter)
            cdot is hadamard
            u is a learnable weight matrix of size (n_out, c_in)
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
            A_options (str):
                root:
                    'none': not learnable
                    'sparse': only the values are learnable
                normalization:
                    To normalize A by degree, suffix with "_norm"
                prefix:
                    N/A
        """
        super().__init__()
        assert(bias_type in ['', 'c', 'nc'])
        normalize = A_options.endswith("_norm")
        if normalize:
            A_options = A_options[:-5]
        use_mask_weight = A_options in ["sparse"]
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        self.maxD = max(len(al) for al in adj_list)

        # parameters
        # self.mask_weight ... initialized based on density
        # For optimization, we reorder the parameter u (self.weight)
        self.weight = nn.Parameter((np.sqrt(2 / outn)) * (-1 + 2 * torch.rand(outn, 1, inc)).float())  # u
        self.channel_transform = weight_norm(nn.Linear(inc, outc))  # v
        if bias_type == '':
            bias = torch.zeros(outc, 1).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, 1).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, outn).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

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
            self.density = (mask > 0).sum().item() / np.prod(mask.shape)
            if ((not(must_use_padded)) and (self.density <= DENSITY_THRESHOLD)):  # Arbitrary threshold
                print("FGL_v2 Optimization: Using packed. density={}".format(self.density))
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
                if use_mask_weight:
                    self.mask_weight_packed = nn.Parameter(torch.randn((int(lengths.sum().item()), 1), dtype=torch.float))

                def forward_(self, x):
                    # x: N, inc, inn
                    # import pdb; pdb.set_trace()
                    N = x.shape[0]
                    embedding_weight = x.view(-1, self.inn).t()  # inn, (N * inc)
                    embedding_output = F.embedding(self.A_packed_data, embedding_weight)  # L, (N * inc)
                    masked_embedding_output, _ = rnn.pad_packed_sequence(
                        rnn.PackedSequence(
                            data=self.mask_weight_packed * embedding_output if use_mask_weight else embedding_output,
                            batch_sizes=self.A_packed_batch_sizes,
                        ),
                        batch_first=True,
                    )  # outn, D, (N * inc)
                    pooled_masked_embedding_output = masked_embedding_output.sum(dim=1)  # outn, (N * inc) but in wrong order.
                    gathered_pooled_masked_embedding_output = pooled_masked_embedding_output[self.original_idx]  # outn, (N * inc) in correct order.
                    scaled_gpmeo = gathered_pooled_masked_embedding_output * self.weight.expand(self.outn, N, self.inc).contiguous().view(self.outn, N * self.inc)
                    pre_channel_transform = scaled_gpmeo.view(-1, self.inc)  # .view(self.outn, N, self.inc).contiguous()  # Flattened for linear
                    almost_y = self.channel_transform(pre_channel_transform).view(self.outn, N, self.outc)  # outn, N, outc
                    y = almost_y.contiguous().permute(1, 2, 0).contiguous()
                    return y + self.bias

                # self.get_almost_y = self.packed_forward
            else:  # Padded
                # Used as input to embedding matrix.
                self.register_buffer('A', A)
                if use_mask_weight:
                    self.mask_weight = nn.Parameter(torch.randn((outn, self.maxD, 1), dtype=torch.float))

                def forward_(self, x):
                    # x: N, inc, inn
                    N = x.shape[0]
                    embedding_weight = x.view(-1, self.inn).t()  # .contiguous?
                    embedding_output = F.embedding(self.A, embedding_weight)  # outn, maxD, (N * inc)
                    masked_embedding_output = self.mask_weight * self.mask * embedding_output if use_mask_weight else self.mask * embedding_output  # outn, maxD, N * inc
                    pooled_masked_embedding_output = masked_embedding_output.sum(dim=1).view(-1, self.inc)  # .view(self.outn, self.maxD, N, self.inc).sum(dim=1).view(-1, self.inc)  # outn, N, inc
                    scaled_gpmeo = pooled_masked_embedding_output * self.weight.expand(self.outn, N, self.inc).contiguous().view(self.outn * N, self.inc)
                    pre_channel_transform = scaled_gpmeo.view(-1, self.inc)
                    almost_y = self.channel_transform(pre_channel_transform).view(self.outn, N, self.outc)  # outn, N, outc
                    y = almost_y.contiguous().permute(1, 2, 0).contiguous()
                    return y + self.bias

                # self.get_almost_y = self.padded_forward

        self.forward_ = forward_
        mask = mask.unsqueeze(2)
        self.register_buffer('mask', mask)
        self.register_buffer('lengths', lengths)

    def forward(self, x):
        # x: N, inc, inn
        return self.forward_(self, x)


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
    def __init__(self, inc, inn, outc, outn, dict_adj_lists, A_options='none_norm', reduction='sum', bias_type='nc', use_spectral_norm=False):
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
            str(k): maybe_spec_norm(FGL(inc, inn, outc, outn, v, A_options=A_options, bias_type=bias_type)) for k, v in dict_adj_lists.items()
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


versions = {
    0: FGL_useless,
    1: FGL,
    2: FGL_v2,
}


if __name__ == '__main__':
    import pdb
    # FGL = FGL_v2
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
        print("density = {}".format(fgl.density))
        x = torch.randn(N, cin, nin).cuda()
        y = fgl(x)
        x = torch.randn(N, cin, nin).cuda()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = fgl(x)
        # print(prof)

    def profiling_test(must_use_padded, tree_optimization=False):
        N = 32
        nin = 32768
        cin = 64
        nout = 2 * 8192
        cout = 128
        adj_list = [[i] for i in range(nout)]  # Non-empty!
        np.random.seed(1337)
        for i in range(nout, nin):  # Tree
            parent = np.random.randint(0, nout)
            if nin not in adj_list[parent]:  # Avoid duplicates.
                adj_list[parent].append(i)
        fgl = FGL(cin, nin, cout, nout, adj_list, must_use_padded=must_use_padded, tree_optimization=tree_optimization).cuda()
        print("density = {}".format(fgl.density))
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

    print("Using tree_optimization:")
    start = time.time()
    profiling_test(False, tree_optimization=True)
    tree_time = time.time() - start

    print("Packed={} Padded={} Tree={}".format(packed_time, padded_time, tree_time))
    pdb.set_trace()
