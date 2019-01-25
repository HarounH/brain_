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


class RandomFGLClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size]
        self.channel_sizes = [1, 32, 64, 128]  # That mapping should be fairly fast

        cur_count = self.node_sizes[0]
        adj_list = []
        for next_count in self.node_sizes[1:]:
            adj_list.append(utils.random_tree_adjacency_list(cur_count, next_count))
            cur_count = next_count

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
        self.activation2 = nn.Sequential()
        self.fc = nn.Sequential(
            weight_norm(nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['c2i']))),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


class Classifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size]
        self.channel_sizes = [1, 32, 64, 128]  # That mapping should be fairly fast

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
        self.activation2 = nn.Sequential()
        self.fc = nn.Sequential(
            weight_norm(nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['c2i']))),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


class ResidualClassifier(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size * 128, z_size]
        self.channel_sizes = [1, 32, 64, 64, 128]  # That mapping should be fairly fast

        adj_list = []
        cur_level = wtree.get_leaves()
        cur_level, _, adj = wtree.get_level_and_adjacency(z_size * 512, cur_level)
        adj_list.append(adj)
        cur_level, _, adj = wtree.get_level_and_adjacency(z_size * 128, cur_level)
        adj_list.append(adj)
        adj_list.append(wtree.get_self_adj(cur_level))
        cur_level, _, adj = wtree.get_level_and_adjacency(z_size, cur_level)
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
            optimization="packed0.3",
        )
        self.activation2 = nn.Sequential()
        self.downsample3 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[3]),
            int(self.node_sizes[3]),
            int(self.channel_sizes[4]),
            int(self.node_sizes[4]),
            adj_list[3],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.activation3 = nn.Sequential()
        self.fc = nn.Sequential(
            weight_norm(nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['c2i']))),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        cur_z = self.activation0(self.downsample0(cur_z))
        cur_z = self.activation1(self.downsample1(cur_z))
        residual = self.activation2(self.downsample2(cur_z))
        cur_z = self.activation3(self.downsample3(cur_z + residual))
        return self.fc(cur_z.view(N, -1))
