"""
For the FGL version that uses op_order etc.
"""

from collections import defaultdict
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import (
    spectral_norm,
    weight_norm,
)
from gcn.modules import fgl
import data.constants as constants
import data.ward_tree as ward_tree
import utils.utils as utils


class SmallClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=32, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        self.node_sizes = [in_features, 4096, 1024, 64]
        self.channel_sizes = [1, 32, 64, 128]  # That mapping should be fairly fast
        adj_list = []

        cur_level = wtree.get_leaves()
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = wtree.get_level_and_adjacency(next_count, cur_level)
            adj_list.append(adj)
        # adj_list contains adj list from ~200k->...->128
        # we need to transpose each one and them reverse the list
        self.n_layers = len(self.channel_sizes) - 1
        OP_ORDER = args.op_order  # "132"
        REDUCTION = args.reduction
        OPTIMIZATION = args.optimization
        self.downsample0 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[0]),
            int(self.node_sizes[0]),
            int(self.channel_sizes[1]),
            int(self.node_sizes[1]),
            adj_list[0],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation0 = nn.Sequential()
        self.downsample1 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[1]),
            int(self.node_sizes[1]),
            int(self.channel_sizes[2]),
            int(self.node_sizes[2]),
            adj_list[1],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation1 = nn.Sequential()
        self.downsample2 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[2]),
            int(self.node_sizes[2]),
            int(self.channel_sizes[3]),
            int(self.node_sizes[3]),
            adj_list[2],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation2 = nn.Dropout() if args.dropout else nn.Sequential()

        fcs = []
        for i, study in enumerate(args.studies):
            si = args.meta['s2i'][study]
            assert(i == si)
            nclasses = len(args.meta['si2ci'][si])
            fcs.append(
                weight_norm(nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], nclasses))
            )
        self.fcs = nn.ModuleList(fcs)

    def forward(self, study_vec, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fcs[study_vec[0].item()](cur_z.view(N, -1))



class SmallerClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=32, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        self.node_sizes = [in_features, 1024, 256, 32]
        self.channel_sizes = [1, 32, 64, 128]  # That mapping should be fairly fast
        # self.channel_sizes = [1, 2, 4, 8]  # That mapping should be fairly fast
        adj_list = []

        cur_level = wtree.get_leaves()
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = wtree.get_level_and_adjacency(next_count, cur_level)
            # cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from ~200k->...->128
        # we need to transpose each one and them reverse the list
        self.n_layers = len(self.channel_sizes) - 1
        OP_ORDER = args.op_order  # "132"
        REDUCTION = args.reduction
        OPTIMIZATION = args.optimization
        self.downsample0 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[0]),
            int(self.node_sizes[0]),
            int(self.channel_sizes[1]),
            int(self.node_sizes[1]),
            adj_list[0],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation0 = nn.Sequential()
        self.downsample1 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[1]),
            int(self.node_sizes[1]),
            int(self.channel_sizes[2]),
            int(self.node_sizes[2]),
            adj_list[1],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation1 = nn.Sequential()
        self.downsample2 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[2]),
            int(self.node_sizes[2]),
            int(self.channel_sizes[3]),
            int(self.node_sizes[3]),
            adj_list[2],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation2 = nn.Dropout() if args.dropout else nn.Sequential()

        fcs = []
        for i, study in enumerate(args.studies):
            si = args.meta['s2i'][study]
            assert(i == si)
            nclasses = len(args.meta['si2ci'][si])
            fcs.append(
                weight_norm(nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], nclasses))
            )
        self.fcs = nn.ModuleList(fcs)

    def forward(self, study_vec, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fcs[study_vec[0].item()](cur_z.view(N, -1))
