"""
FGL version of encoder.
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


class FGLEncoder0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        in_features = constants.original_masked_nnz

        # Single layer downsampling.
        # This set up ensures that ConvEncoder and FGL Encoder have same output size.
        self.node_sizes = [in_features, 512, 32]
        self.channel_sizes = [1, 32, 128]  # That mapping should be fairly fast

        # self.node_sizes = [in_features, z_size * 512, z_size, 12]  # , z_size * 512, z_size * 128, z_size]
        # self.channel_sizes = [1, z_size // 16, z_size // 4, z_size]  # , 32, 64, 128]  # That mapping should be fairly fast

        cur_level = wtree.get_leaves()
        adj_list = []
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = wtree.get_level_and_adjacency(next_count, cur_level, n_regions=2)
            adj_list.append(adj)

        self.n_layers = len(self.channel_sizes) - 1
        OP_ORDER = "132" # args.op_order
        REDUCTION = "sum" # args.reduction
        OPTIMIZATION = "tree"  # args.optimization
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
        # self.downsample2 = fgl.make_weight_normed_FGL(
        #     int(self.channel_sizes[2]),
        #     int(self.node_sizes[2]),
        #     int(self.channel_sizes[3]),
        #     int(self.node_sizes[3]),
        #     adj_list[2],
        #     op_order=OP_ORDER,
        #     reduction=REDUCTION,
        #     optimization=OPTIMIZATION,
        # )
        self.linear = nn.Linear(self.channel_sizes[-1] * self.node_sizes[-1], 6 * 128)

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
        return self.linear(cur_z.view(N, -1))
