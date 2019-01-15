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
    weight_norm,
    rnn,
)
import utils.utils as utils
import torch_scatter


DENSITY_THRESHOLD = 0.25  # If sparse adjacency matrix, A, is less occupied than this, then packed tensors are used instead of padded for embedding (which is the bottleneck)


class FGL(nn.Module):
    def __init__(self, inc, inn, outc, outn, adj_list, op_order='132', reduction='max', bias_type='nc', optimization='packed0.3'):
        """
            Does three things:
                1. Node/Feature weights: parameter of size (n * c)
                2. Feature map application: parameter of size (inc * outc)
                3. Node reduction (no parameters)
            (and also adds the bias, but whatever).
            In a fully connected net, 1, 2, 3 would be combined into one - leading to
            a large parameter of size (inn * outn * inc * outc).
            This is infeasible, and the kind of split proposed by FGL is doable.

            Args:
                inc (int): Input channels
                inn (int): Input positions
                outc (int): Output channels
                outn (int): Output positions
                adj_list (int list list):
                    adjacency list, from output nodes to input nodes.

                op_order (str): string determining the variant of FGL
                    some permutation of 1, 2, 3 - determines the order in which 3 steps are done.

                reduction (str): Determines how reduction over nodes in each output node is done
                    "sum":
                    "learn": A sum reduction with learnt weights.
                    "mean":
                    "max":

                bias_type (str):
                    'none': no bias is used
                    'c': a bias for each channel (same over output nodes)
                    'nc': a bias for each channel and output node.

                optimization (str): determines how the node-reduction operation is implemented.
                    "none": Good old padded adjacency list is used
                    "tree": if the adjacency list comes from a tree, an optimization using scatter_all_ is used
                    "packed[0-9\.]*":
                        If the density of A is greater than the number specified (if no number, then a default is used),
                        then, a packed representation is used instead of padded.
                        For brain-decoding, A usually has a density < 25%, so this can speed up the embedding stage a LOT!
        """
        super().__init__()
        assert(len(op_order) == 3)
        assert(reduction in ["sum", "mean", "max", "learn"])
        self.reduction = reduction
        assert(bias_type in ['', 'c', 'nc'])
        self.inc = inc
        self.outc = outc
        self.inn = inn
        self.outn = outn
        lengths = torch.tensor([len(al) for al in adj_list]).long()
        lengths_f32 = lengths.float().unsqueeze(1)
        self.register_buffer('lengths', lengths)
        self.register_buffer('lengths_f32', lengths_f32)
        self.maxD = lengths.max().item()
        self.op_order = op_order

        # bias is easy
        if bias_type == '':
            bias = torch.zeros(outc).float()
            self.register_buffer('bias', bias)
        elif bias_type == 'c':
            self.bias = nn.Parameter(0.2 * torch.randn(outc).float())
        elif bias_type == 'nc':
            self.bias = nn.Parameter(0.2 * torch.randn(outc, outn).float())
        else:
            raise NotImplementedError("Bias type {} not implemented".format(bias_type))

        # Feature map... never changes.
        self.ft_weight = nn.Parameter((-1.0 + 2 * torch.rand(inc, outc)) * np.sqrt(6.0 / (1.0 + 5 * inc)))

        def feature_transform_(self, x):
            # x: N, inc, *
            # output: N, outc, *
            # Hallelujah PyTorch 1.0!
            return torch.einsum("...ij,ik->...kj", x, self.ft_weight).contiguous()

        # Node/Feature weight.
        c_ = inc if op_order.find("2") > op_order.find("1") else outc
        n_ = inn if op_order.find("3") > op_order.find("1") else outn
        self.nf_weight = nn.Parameter(torch.randn(c_, n_))

        def nf_transform_(self, x):
            # x: N, c_, n_
            # output: N, c_, n_
            return (x * self.nf_weight).contiguous()

        self.func_map = {
            "1": nf_transform_,
            "2": feature_transform_,
            "3": None,  # Set based on optimization
        }
        # Reduction
        if optimization == "tree":
            print("FGL: Using tree optimization")
            indices = torch.tensor(utils.transpose_adj_list(self.outn, self.inn, adj_list)).long().t().unsqueeze(0)
            self.register_buffer('indices', indices)

            # Hallelujah torch-scatter
            if reduction == "sum":
                self.scatter_func_ = torch_scatter.scatter_add
            elif reduction == "mean":
                self.scatter_func_ = torch_scatter.scatter_mean
            elif reduction == "max":
                self.scatter_func_ = torch_scatter.scatter_max
            else:
                raise NotImplementedError("Cant do {} reduction with {} optimization".format(reduction, optimization))

            def reduction_(self, x):
                # x: N, *, inn
                # output: N, *, outn
                y = torch.zeros(x.shape[0], x.shape[1], self.outn, dtype=torch.float, device=x.device)
                self.scatter_func_(x, self.indices.to(x.device).expand(x.shape), dim=2, out=y)
                return y

            self.func_map["3"] = reduction_
        else:
            mask = torch.zeros(outn, self.maxD).float()  # Binary mask
            padded_adj_list = []
            for nidx, al in enumerate(adj_list):
                mask[nidx, :len(al)] = 1.0
                padded_adj_list.append(al + [0 for i in range(self.maxD - len(al))])
            A = torch.tensor(padded_adj_list).long()  # N, T
            self.density = mask.sum().item() / np.prod(mask.shape)
            self.total_length = self.lengths.sum().item()

            if (optimization != "none") and (self.density > float(optimization[6:])):
                print("FGL: Using padded variant: density={}".format(self.density))
                self.register_buffer('A', A)
                if reduction == "learn":
                    self.mask_weight = nn.Parameter(torch.randn_like(mask, dtype=torch.float))
                else:
                    self.mask_weight = 1.0

                def reduction_(self, x):
                    # Padded embedding and stuff.
                    # x: N, _, inn
                    # y: N, _, outn
                    N, c, n = x.shape
                    embw = x.view(-1, n).t()
                    embo = F.embedding(self.A, embw)  # outn, maxD, (N * c)
                    masked_embo = self.mask_weight * self.mask * embo  # outn, maxD, (N * c)
                    if self.reduction == "max":
                        pooled_masked_embo = masked_embo.max(dim=1)[0]
                    else:
                        pooled_masked_embo = masked_embo.sum(dim=1)
                        if self.reduction == "mean":
                            pooled_masked_embo = pooled_masked_embo / self.lengths_f32
                    return pooled_masked_embo.t().view(N, c, -1).contiguous()
                self.func_map["3"] = reduction_
            else:
                print("FGL: Using packed variant: density={}".format(self.density))

                sorted_len, sorted_idx = lengths.sort(0, descending=True)
                _, original_idx = sorted_idx.sort(0, descending=True)
                A_packed = rnn.pack_padded_sequence(A[sorted_idx, :], sorted_len, batch_first=True)

                self.register_buffer('A_packed_data', A_packed.data)
                self.A_packed_batch_sizes = A_packed.batch_sizes
                # self.register_buffer('A_packed_batch_sizes', A_packed.batch_sizes)
                self.register_buffer('sorted_len', sorted_len)
                self.register_buffer('sorted_idx', sorted_idx)
                self.register_buffer('original_idx', original_idx)  # Enable CUDA transfer

                if reduction == "learn":
                    self.mask_weight_packed = nn.Parameter(torch.randn((int(lengths.sum().item()), 1), dtype=torch.float))
                else:
                    self.mask_weight_packed = 1.0

                def reduction_(self, x):
                    # x: N, c, inn
                    # output: N, c, outn
                    N, c, n = x.shape
                    embw = x.view(-1, n).t()
                    embo = F.embedding(self.A_packed_data, embw)  # L, (N * c)
                    masked_embo, _ = rnn.pad_packed_sequence(
                        rnn.PackedSequence(
                            data=self.mask_weight_packed * embo,
                            batch_sizes=self.A_packed_batch_sizes,
                        ),
                        batch_first=True,
                    )  # outn, D, (N * c)
                    if self.reduction == "max":
                        pooled_masked_embo = masked_embo.max(dim=1)[0]
                    else:
                        pooled_masked_embo = masked_embo.sum(dim=1)
                        if self.reduction == "mean":
                            pooled_masked_embo = pooled_masked_embo / self.lengths_f32
                    return pooled_masked_embo.t().view(N, c, -1).contiguous()

                self.func_map["3"] = reduction_

    def forward(self, x):
        y = self.func_map[self.op_order[0]](self, x)
        y = self.func_map[self.op_order[1]](self, y)
        y = self.func_map[self.op_order[2]](self, y)
        return y + self.bias


if __name__ == '__main__':
    # Code to profile FGL
    import time
    import itertools

    N = 32
    nin = 32768
    cin = 64
    nout = 2 * 8192
    cout = 128
    adj_list = [[i] for i in range(nout)]
    np.random.seed(1337)
    for i in range(nout, nin):
        parent = np.random.randint(0, nout)
        if nin not in adj_list[parent]:  # Avoid duplicates.
            adj_list[parent].append(i)
    x = torch.randn(N, cin, nin).cuda()

    def run(cin, nin, cout, nout, adj_list, x, return_profile, op_order='123', reduction='mean', bias_type='nc', optimization='none'):
        k = "/".join([op_order, bias_type, reduction, optimization])
        print("Starting {}".format(k))
        fgl = FGL(cin, nin, cout, nout, adj_list).cuda()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            y = fgl(x)
        return prof

    op_orders = ["123", "132", "213", "231"]  # ["".join(x) for x in itertools.permutations(["1", "2", "3"], 3)]
    bias_types = ["c"]  # ["", "c", "nc"]
    reductions = ["mean", "max"]
    optimizations = ["none", "tree", "packed0.3"]
    # Warm up
    run(cin, nin, cout, nout, adj_list, x, False)
    run(cin, nin, cout, nout, adj_list, x, False)
    # Start profiling
    times = {}
    profs = {}
    all_combos = itertools.product(op_orders, bias_types, reductions, optimizations)
    print("Profiling {} combinations".format(all_combos))
    for (op_order, bias_type, reduction, optimization) in all_combos:
        k = "/".join([op_order, bias_type, reduction, optimization])
        # print("Starting {}".format(k))
        tic = time.time()
        profs[k] = run(cin, nin, cout, nout, adj_list, x, True, op_order=op_order, reduction=reduction, bias_type=bias_type, optimization=optimization)
        times[k] = time.time() - tic

    for k, v in profs.items():
        print("Prof {}".format(k))
        print(v)

    for k, v in times.items():
        print("{}: {}s".format(k, v))
    import pdb; pdb.set_trace()
