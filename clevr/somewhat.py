'''
A dataset of images containing shapes whose position
is correlated with the shape type.
'''

import numpy as np
import skimage.draw as skdraw
import torch
from torch.utils.data.dataset import Dataset as TorchDataset


class SomewhatClevr(TorchDataset):
    def __init__(self):
        super().__init__()
        self.order = ['circle', 'square']

        # Quadrant-like
        self.s = s = 128
        self.l = l = 9
        self.r = r = 5

        self.square_r = lambda x: x + np.array([- (l // 2), - (l // 2), l // 2, l // 2])
        self.square_c = lambda y: y + np.array([-(l // 2), l // 2, l // 2, -(l // 2)])

        ypos, xpos = np.meshgrid(list(range(s)), list(range(s)))
        xypos = np.stack([xpos, ypos], axis=-1)

        quadrants_dict = {
            0: np.logical_and(xpos < s//2, ypos < s//2),
            1: np.logical_and(xpos >= s//2, ypos < s//2),
            2: np.logical_and(xpos >= s//2, ypos >= s//2),
            3: np.logical_and(xpos < s//2, ypos >= s//2),
        }

        self.quadrants = sum(i * v.astype(np.float) for i, v in quadrants_dict.items())

        # circle is more likely in quadrant 2 and less likely in quadrant 0.
        # square is more likely in quadrant 0 and less in quadrant 2.
        # This
        # Shapes have to present entirely within a quadrant for this to make sense.
        # offset padding = 3
        offpad = 6
        valid_center = {
            'circle': {
                0: np.logical_and(
                    quadrants_dict[0],
                    np.logical_and(
                        np.logical_and(xpos > r - 1 + offpad, xpos < s // 2 - r + 1 - offpad),
                        np.logical_and(ypos > r - 1 + offpad, ypos < s // 2 - r + 1 - offpad),
                    )
                ),
                1: np.logical_and(
                    quadrants_dict[1],
                    np.logical_and(
                        np.logical_and(xpos > s // 2 + r - 1 + offpad // 2, xpos < s - r + 1 - offpad // 2),
                        np.logical_and(ypos > r - 1 + offpad // 2, ypos < s // 2 - r + 1 - offpad // 2),
                    )
                ),
                2: np.logical_and(
                    quadrants_dict[2],
                    np.logical_and(
                        np.logical_and(xpos > s // 2 + r - 1, xpos < s - r + 1),
                        np.logical_and(ypos > s // 2 + r - 1, ypos < s - r + 1),
                    )
                ),
                3: np.logical_and(
                    quadrants_dict[3],
                    np.logical_and(
                        np.logical_and(xpos > r - 1 + offpad // 2, xpos < s // 2 - r + 1 - offpad // 2),
                        np.logical_and(ypos > s // 2 + r - 1 + offpad // 2, ypos < s - r + 1 - offpad // 2),
                    )
                ),
            },
            'square': {
                0: np.logical_and(
                    quadrants_dict[0],
                    np.logical_and(
                        np.logical_and(xpos > l // 2, xpos < s // 2 - l // 2),
                        np.logical_and(ypos > l // 2, ypos < s // 2 - l // 2),
                    )
                ),
                1: np.logical_and(
                    quadrants_dict[1],
                    np.logical_and(
                        np.logical_and(xpos > s // 2 + l // 2 + offpad // 2, xpos < s - l // 2 - offpad // 2),
                        np.logical_and(ypos > l // 2 + offpad // 2, ypos < s // 2 - l // 2 - offpad // 2),
                    )
                ),
                2: np.logical_and(
                    quadrants_dict[2],
                    np.logical_and(
                        np.logical_and(xpos > s // 2 + l // 2 + offpad, xpos < s - l // 2 - offpad),
                        np.logical_and(ypos > s // 2 + l // 2 + offpad, ypos < s - l // 2 - offpad),
                    )
                ),
                3: np.logical_and(
                    quadrants_dict[3],
                    np.logical_and(
                        np.logical_and(xpos > l // 2 + offpad // 2, xpos < s // 2 - l // 2 - offpad // 2),
                        np.logical_and(ypos > s // 2 + l // 2 + offpad // 2, ypos < s - l // 2 - offpad // 2),
                    )
                ),
            },
        }

        self.centers = {
            sh: xypos[
                np.logical_or(
                    np.logical_or(valid_center[sh][0], valid_center[sh][1]),
                    np.logical_or(valid_center[sh][2], valid_center[sh][3])
                )
            ]
            for sh in valid_center.keys()
        }
        self.n = self.centers['square'].shape[0]
        self.n_classes = 8
    def __len__(self):
        return self.n * 2

    def __getitem__(self, idx):
        img = np.zeros((self.s, self.s))
        off = 0
        if idx < self.n:  # Square
            x, y = self.centers['square'][idx]
            sqr, sqc = self.square_r(y), self.square_c(x)
            rr, cc = skdraw.polygon(sqr, sqc)
        else:
            off = 4
            x, y = self.centers['circle'][idx - self.n]
            rr, cc = skdraw.circle(x, y, self.r)

        img[rr, cc] = 1.0
        return img, off + int(self.quadrants[x, y])


if __name__ == '__main__':
    # Generate a dataset
    dset = SomewhatClevr()
    import pdb; pdb.set_trace()
