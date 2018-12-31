from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gcn.modules.fgl as fgl
import utils.utils as utils
import data.ward_tree as ward_tree


class FGLGeneratorHierarchical0(nn.Module):
    def __init__(self, args, wtree, loadable_state_dict=None, z_size=128, content_channels=16, dropout_rate=0.5):
        super(FGLGeneratorHierarchical0, self).__init__()
        # wtree: WardTree object.
        # arch: N, 128 (z_size + cc) -> N, 512 (z_size // 2 + cc) -> N, 2048 (32) -> N, 8192 (16) -> N, 32768 (8) -4-> N, 67615 (1)
        self.args = args
        meta = self.args.meta

        # FCs
        self.study_embedding = weight_norm(nn.Embedding(len(meta['s2i']), content_channels))
        self.task_embedding = weight_norm(nn.Embedding(len(meta['t2i']), content_channels))
        self.contrast_embedding = weight_norm(nn.Embedding(len(meta['c2i']), content_channels))
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(2 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
            nn.Sequential(
                nn.Linear(3 * content_channels, content_channels),
                nn.Dropout(dropout_rate),
            ),
        ])

        self.nodes_arr = [32768, 8192, 2048, 512, 128]
        adjes = []
        cur_level = wtree.get_leaves()
        for next_count in self.nodes_arr:
            cur_level, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adjes.append(adj)
            cur_c = next_c
        adjes = adjes[::-1]

        self.upsample0 = fgl.FGL(128 + content_channels, 64, utisl.scsp2tsp(adjes[0].T))
        self.activation0 = nn.Sequential(nn.LeakyReLU(0.2), )
        self.upsample1 = fgl.FGL(64 + content_channels, 32, utisl.scsp2tsp(adjes[1].T))
        self.upsample2 = fgl.FGL(32 + content_channels, 16, utisl.scsp2tsp(adjes[2].T))
        self.upsample3 = fgl.FGL(16 + content_channels, 8, utisl.scsp2tsp(adjes[3].T))
        self.upsample4 = fgl.FGL(8 + content_channels, 1, utisl.scsp2tsp(adjes[4].T))

    def forward(self, z):
        # z: N, latent_dim
        z = z.unsqueeze(1).expand(z.shape[0], self.nodes_arr[-1], z.shape[1])

        se = self.study_embedding(studies)
        te = self.task_embedding(tasks)
        ce = self.contrast_embedding(contrasts)
        contents = [
            se,
            torch.cat([se, te], dim=1),  # se + te,
            torch.cat([se, te, ce], dim=1),  # se + te + ce
        ]
        contents.append(contents[-1])
        contents.append(contents[-1])
        contents = [self.fcs[i](content) for i, content in enumerate(contents)]
        for i in range(5):
            upsample = getattr(self, 'upsample{}'.format(i))




versions = {
    '0': FGLGeneratorHierarchical0,
}
