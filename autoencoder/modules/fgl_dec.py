"""
FGL version of decoder.
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


class FGLDecoder0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size
        out_features = constants.original_masked_nnz

        # self.node_sizes = [out_features, z_size * 512, z_size, 12]  # , z_size * 512, z_size * 128, z_size]
        # self.channel_sizes = [1, z_size // 16, z_size // 4, z_size]  # , 32, 64, 128]  # That mapping should be fairly fast

        self.node_sizes = [out_features, 512, 32]
        self.channel_sizes = [1, 32, 128]  # That mapping should be fairly fast

        cur_level = wtree.get_leaves()
        adj_list = []
        for next_count in self.node_sizes[1:]:
            cur_count = len(cur_level)
            cur_level, _, adj = wtree.get_level_and_adjacency(next_count, cur_level, n_regions=2)
            adj_list.append(utils.transpose_adj_list(next_count, cur_count, adj))

        self.n_layers = len(self.channel_sizes) - 1
        OP_ORDER = "213"  # args.op_order
        REDUCTION = "sum" # args.reduction
        OPTIMIZATION = "packed1.0" # args.optimization
        self.upsample0 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[-1]),
            int(self.node_sizes[-1]),
            int(self.channel_sizes[-2]),
            int(self.node_sizes[-2]),
            adj_list[-1],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        self.upsample1 = fgl.make_weight_normed_FGL(
            int(self.channel_sizes[-2]),
            int(self.node_sizes[-2]),
            int(self.channel_sizes[-3]),
            int(self.node_sizes[-3]),
            adj_list[-2],
            op_order=OP_ORDER,
            reduction=REDUCTION,
            optimization=OPTIMIZATION,
        )
        # self.upsample2 = fgl.make_weight_normed_FGL(
        #     int(self.channel_sizes[-3]),
        #     int(self.node_sizes[-3]),
        #     int(self.channel_sizes[-4]),
        #     int(self.node_sizes[-4]),
        #     adj_list[-3],
        #     op_order=OP_ORDER,
        #     reduction=REDUCTION,
        #     optimization=OPTIMIZATION,
        # )
        self.linear = nn.Linear(6 * 128, self.channel_sizes[-1] * self.node_sizes[-1])

    def forward(self, x):
        # x: N, emb-size
        N = x.shape[0]
        cur_z = self.linear(x).unsqueeze(1).view(N, self.channel_sizes[-1], self.node_sizes[-1])
        for i in range(self.n_layers):
            cur_z = getattr(self, 'upsample{}'.format(i))(cur_z)
        return cur_z.view(N, -1)
