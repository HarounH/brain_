''' A dataset containing images cut up into wedges/sectors.
'''
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset as TorchDataset
try:
    import cairo
except:
    print('cairo was not found. ignoring and moving on.')

class GradientWedges(TorchDataset):
    def __init__(self, args, s=64, r=24, n=50000):
        super().__init__()
        self.s = s
        self.sb2 = s // 2
        self.r = r
        self.n = n
        self.pibnc = np.pi / 2

        self.pattern_generators = [
            # lambda: self.make_random_pattern(),
            # lambda: self.make_radial_pattern(),
            lambda: self.make_horizontal_linear_pattern(),
            lambda: self.make_vertical_linear_pattern(),
        ]
        self.n_pattern_types = len(self.pattern_generators)  # random, radial, horizontal linear, vertical linear
        self.regions = self.init_regions()
        self.n_regions = len(self.regions)
        self.n_classes = self.n_regions * self.n_pattern_types

        # X noise generation parameters
        self.general_noise = 0.5  # Dampened noise.
        self.wedge_noise = 1.0
        self.pattern_alpha_cut = 1.5

        # self.wedge_bias = 0.5
        self.Y = list(range(self.n_classes))
        self.X = self.generate_x(args.dset_seed)

    def make_random_pattern(self):
        arr = self.wedge_noise * np.random.rand(self.s, self.s).astype(np.float32)
        return arr

    def make_radial_pattern(self):
        s = self.s
        arr = np.zeros((self.s, self.s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            arr, cairo.FORMAT_ARGB32, self.s, self.s)
        cr = cairo.Context(surface)
        r1 = cairo.RadialGradient(self.s // 2, self.s // 2, self.r // 4, self.s // 2, self.s // 2, 2 * self.r)
        val = np.random.rand() / self.pattern_alpha_cut  # 0->1.0
        r1.add_color_stop_rgba(0, 1, 0, 0, val)
        r1.add_color_stop_rgba(1, 0, 1, 0, 1 - val)
        cr.set_source(r1)
        cr.move_to(s//2, s//2)
        cr.arc(s//2, s//2, s, 0, 2 * np.pi)
        cr.close_path()
        cr.fill()
        return arr[:, :, 3].astype(np.float32) / 255

    def make_horizontal_linear_pattern(self):
        s = self.s
        arr = np.zeros((self.s, self.s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            arr, cairo.FORMAT_ARGB32, self.s, self.s)
        cr = cairo.Context(surface)
        r1 = cairo.LinearGradient(0, self.s // 2, self.s, self.s // 2)
        val = np.random.rand() / self.pattern_alpha_cut  # 0->1.0
        r1.add_color_stop_rgba(0, 1, 0, 0, val)
        r1.add_color_stop_rgba(1, 0, 1, 0, 1 - val)
        cr.set_source(r1)
        cr.move_to(s//2, s//2)
        cr.arc(s//2, s//2, s, 0, 2 * np.pi)
        cr.close_path()
        cr.fill()
        return arr[:, :, 3].astype(np.float32) / 255

    def make_vertical_linear_pattern(self):
        s = self.s
        arr = np.zeros((self.s, self.s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            arr, cairo.FORMAT_ARGB32, self.s, self.s)
        cr = cairo.Context(surface)
        r1 = cairo.LinearGradient(self.s // 2, 0, self.s // 2, self.s)
        val = np.random.rand() / self.pattern_alpha_cut  # 0->1.0
        r1.add_color_stop_rgba(0, 1, 0, 0, val)
        r1.add_color_stop_rgba(1, 0, 1, 0, 1 - val)
        cr.set_source(r1)
        cr.move_to(s//2, s//2)
        cr.arc(s//2, s//2, s, 0, 2 * np.pi)
        cr.close_path()
        cr.fill()
        return arr[:, :, 3].astype(np.float32) / 255

    def init_regions(self):
        s = self.s
        regions = {}
        for i in range(4):
            x = np.zeros((s, s, 4)).astype(np.uint8)
            surface = cairo.ImageSurface.create_for_data(
                x, cairo.FORMAT_ARGB32, s, s)
            cr = cairo.Context(surface)
            cr.set_source_rgba(0, 0, 0, 1)
            cr.move_to(self.sb2, self.sb2)
            cr.arc(self.sb2, self.sb2, self.r, i * self.pibnc, (i + 1) * self.pibnc)
            cr.close_path()
            cr.fill()
            x[x > 0] = 255
            regions[i + 1] = x.astype(np.float32)[:, :, 3] / 255

        x = np.zeros((s, s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            x, cairo.FORMAT_ARGB32, s, s)
        cr = cairo.Context(surface)
        cr.set_source_rgba(0, 0, 0, 1.0)
        cr.move_to(self.sb2, self.sb2)
        cr.arc(self.sb2, self.sb2, self.r, 0, 2 * np.pi)
        cr.close_path()
        cr.fill()
        regions[0] = (x[:, :, 3] == 0).astype(np.float32)
        return regions

    def generate_x(self, seed):
        tic = time.time()
        np.random.seed(seed)
        xs = []
        s = self.s

        for idx in range(self.n):
            y = self.Y[idx % self.n_classes]
            region_type = y % self.n_regions
            pattern_type = y // self.n_regions
            region = self.regions[region_type]
            nonregion = (region < 0.5).astype(np.float32)
            x = self.general_noise * nonregion * np.random.rand(s, s).astype(np.float32)
            pattern = self.pattern_generators[pattern_type]()
            x += region * pattern
            xs.append(x.astype(np.float32))
        print("Took {}s to generate X".format(time.time() - tic))
        return xs

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx % self.n_classes]
        x = self.X[idx]
        return x, y


class ComplexWedges(TorchDataset):
    def __init__(self, args, s=128, r=48, n=10000):
        super().__init__()
        self.s = s
        self.sb2 = s // 2
        self.r = r
        self.n = n
        self.n_classes = 8
        self.pibnc = np.pi / self.n_classes
        # X noise generation parameters
        self.general_noise = 0.2
        self.wedge_noise = 1.0
        self.wedge_alphas = [1.0, 0.5]
        # self.wedge_bias = 0.5
        self.Y = list(range(self.n_classes))
        self.X = self.generate_x(args.dset_seed)

    def generate_x(self, seed):
        tic = time.time()
        np.random.seed(seed)
        xs = []
        s = self.s
        x = np.zeros((s, s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            x, cairo.FORMAT_ARGB32, s, s)
        cr = cairo.Context(surface)
        cr.set_source_rgba(0, 0, 0, 1.0)
        cr.move_to(self.sb2, self.sb2)
        cr.arc(self.sb2, self.sb2, self.r, 0, 2 * np.pi)
        cr.close_path()
        cr.fill()
        non_wedge = (x[:, :, 3] == 0).astype(np.float32)

        for idx in range(self.n):
            y = self.Y[idx % self.n_classes]
            x = np.zeros((s, s, 4)).astype(np.uint8)
            surface = cairo.ImageSurface.create_for_data(
                x, cairo.FORMAT_ARGB32, s, s)
            cr = cairo.Context(surface)
            cr.set_source_rgba(0, 0, 0, self.wedge_alphas[0])
            cr.move_to(self.sb2, self.sb2)
            cr.arc(self.sb2, self.sb2, self.r, y * self.pibnc, (y + 1) * self.pibnc)
            cr.close_path()
            cr.fill()
            x[x > 0] = 255

            cr.set_source_rgba(0, 0, 0, self.wedge_alphas[1])
            cr.move_to(self.sb2, self.sb2)
            cr.arc(self.sb2, self.sb2, self.r, (y + self.n_classes) * self.pibnc, (y + 1 + self.n_classes) * self.pibnc)
            cr.close_path()
            cr.fill()
            x[np.logical_and(x > 0, x < 255)] = 128
            nonx = (x[:, :, 3] == 0).astype(np.float32)
            x = x.astype(np.float32)[:, :, 3] / 255
            x = (self.general_noise * nonx * np.random.randn(*x.shape)) \
                + ((x * (self.wedge_noise * np.random.rand())))
            xs.append(x.astype(np.float32))
        print("Took {}s to generate X".format(time.time() - tic))
        return xs

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx % self.n_classes]
        x = self.X[idx]
        return x, y


class Wedges(TorchDataset):
    def __init__(self, s=128, r=32, n=10000):
        super().__init__()
        self.s = s
        self.sb2 = s // 2
        self.r = r
        self.n = 10000
        self.n_classes = 8
        self.Y = list(range(self.n_classes))
        self.pibnc = 2 * np.pi / self.n_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx % self.n_classes]
        s = self.s
        x = np.zeros((s, s, 4)).astype(np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            x, cairo.FORMAT_ARGB32, s, s)
        cr = cairo.Context(surface)
        cr.set_source_rgba(0, 0, 0, 1.0)
        cr.move_to(self.sb2, self.sb2)
        cr.arc(self.sb2, self.sb2, self.r, y * self.pibnc, (y + 1) * self.pibnc)
        cr.close_path()
        cr.fill()
        x = x.astype(np.float32)[:, :, 3] / 255
        x = x * np.random.rand(*x.shape).astype(np.float32)
        return x, y
