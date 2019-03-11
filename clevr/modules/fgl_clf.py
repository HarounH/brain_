try:
    import cairo
except:
    print('cairo was not found. ignoring and moving on.')
import numpy as np
import torch
from torch import nn
from conv.modules import blocks
from gcn.modules import fgl


def quadrant_adjacency(s):
    quads = np.zeros((s, s)).astype(np.int)
    quads[:s//2, :s//2] = 0
    quads[s//2:, :s//2] = 1
    quads[s//2:, s//2:] = 2
    quads[:s//2, s//2:] = 3
    quads = np.reshape(quads, (s * s, ))
    adj = [[] for _ in range(4)]
    for i in range(len(quads)):
        adj[quads[i]].append(i)
    return adj


def wedge_adjacency(s, r, n_classes=8, diagonally_opposite=False):
    pibnc = np.pi / n_classes * (1 if diagonally_opposite else 2)
    img = np.zeros((s, s), dtype=np.int64)
    adj = [[] for _ in range(n_classes + 1)]
    for c in range(n_classes):
        x = np.zeros((s, s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            x, cairo.FORMAT_ARGB32, s, s)
        cr = cairo.Context(surface)
        cr.set_source_rgba(0, 0, 0, 1.0)
        cr.move_to(s // 2, s // 2)
        cr.arc(s // 2, s // 2, r, c * pibnc, (c + 1) * pibnc)
        cr.close_path()
        cr.fill()
        if diagonally_opposite:
            cr.set_source_rgba(0, 0, 0, 1.0)
            cr.move_to(s // 2, s // 2)
            cr.arc(s // 2, s // 2, r, (c + n_classes) * pibnc, (c + 1 + n_classes) * pibnc)
            cr.close_path()
            cr.fill()
        img[x[:, :, 3] > 0] = c + 1
    img = np.reshape(img, (s * s,))
    for i in range(len(img)):
        adj[img[i]].append(i)
    return adj


def regions_adjacency(regions):
    # regions (dict int-> numpy array)
    adj = []
    region_id2idx = {}
    for region_id, arr in regions.items():
        region_id2idx[region_id] = len(adj)
        adj.append(np.where(np.reshape(arr, (-1,)))[0].tolist())
    return adj, region_id2idx


class RegionClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        super().__init__()
        self.args = args
        n_classes = args.meta['n_classes']
        s = args.meta['s']
        regions = args.meta['regions']
        net = []
        adj, self.rid2idx = regions_adjacency(regions)
        self.adj = adj
        nout = len(adj)
        cout = 4
        net.append(
            fgl.FGL(  #fgl.make_weight_normed_FGL(
                1,
                s * s,
                cout,
                nout,
                adj,
                op_order="213",
                reduction="sum",
                optimization="packed1.0",
            )
        )
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(nout * cout, n_classes))
        if loadable_state_dict is not None:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.view(N, -1).unsqueeze(1)).view(N, -1))


class WedgeClassifier0(nn.Module):
    def __init__(self, args, complex=False, loadable_state_dict=None):
        super().__init__()
        self.args = args
        n_classes = args.meta['n_classes']
        s = args.meta['s']
        r = args.meta['r']
        net = []
        adj = wedge_adjacency(s, r, diagonally_opposite=complex)
        nout = len(adj)
        cout = 4
        net.append(
            fgl.FGL(  #fgl.make_weight_normed_FGL(
                1,
                s * s,
                cout,
                nout,
                adj,
                op_order="213",
                reduction="sum",
                optimization="tree",
            )
        )
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(nout * cout, n_classes))

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.view(N, -1).unsqueeze(1)).view(N, -1))


class QuadClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None):
        super().__init__()
        self.args = args
        n_classes = args.meta['n_classes']
        s = args.meta['s']
        adj = quadrant_adjacency(s)
        nout = len(adj)
        cout = 4
        net = []
        net.append(
            fgl.make_weight_normed_FGL(
                1,
                s * s,
                cout,
                nout,
                adj,
                op_order="213",
                reduction="sum",
                optimization="tree",
            )
        )
        self.net = nn.Sequential(*net)
        self.fc = nn.Sequential(nn.Linear(nout * cout, n_classes))

    def forward(self, x):
        N = x.shape[0]
        return self.fc(self.net(x.view(N, -1).unsqueeze(1)).view(N, -1))
