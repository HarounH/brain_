import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset as TorchDataset


class NotSoClevr(TorchDataset):
    def __init__(self, s=64, l=9):
        self.s = s
        self.l = l
        self.lb2 = lb2 = l // 2
        self.allowed_s = allowed_s = s - 2 * lb2
        self.n = n = (allowed_s) ** 2
        onehots = np.pad(
            np.eye(n).reshape((n, allowed_s, allowed_s, 1)),
            ((0, 0), (lb2, lb2), (lb2, lb2), (0, 0)),
            "constant")
        images_F = F.conv2d(
            torch.from_numpy(onehots).squeeze().unsqueeze(1).float(),
            torch.ones((1, 1, l, l)).float(),
            stride=1,
            padding=lb2).squeeze().numpy()

        self.X = images_F

        ypos, xpos = np.meshgrid(list(range(s)), list(range(s)))
        xypos = np.stack([xpos, ypos], axis=-1)

        quadrants_dict = {
            0: np.logical_and(xpos < s//2, ypos < s//2),
            1: np.logical_and(xpos >= s//2, ypos < s//2),
            2: np.logical_and(xpos >= s//2, ypos >= s//2),
            3: np.logical_and(xpos < s//2, ypos >= s//2),
        }

        quadrants = sum(i * v.astype(np.float) for i, v in quadrants_dict.items()).astype(np.int)
        onehots = onehots[:, :, :, 0].astype(bool)
        self.Y = np.concatenate([quadrants[onehots[i, :, :]] for i in range(n)])
        self.n_classes = 4

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
