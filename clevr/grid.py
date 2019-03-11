''' Datasets containing grid cut up into random parts
'''
import os
import time
import random
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset as TorchDataset
try:
    import cairo
except:
    print('cairo was not found. ignoring and moving on.')
import scipy
import scipy.spatial
import scipy.sparse as sp
import joblib
from math import ceil


def get_Ahx(s, regions):
    Ahx = np.zeros((len(regions), s * s))
    for ridx, cis in regions.items():
        Ahx[ridx, np.reshape(cis, (-1,)) > 0] = 1.0
    return Ahx


def exclusive_regions(seed, s=128, n_regions=8, n_centers=32):
    np.random.seed(seed)
    random.seed(seed)
    regions = np.zeros((s, s)).astype(np.int64)
    xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
    xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
    centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
    distances = scipy.spatial.distance.cdist(xypos, centers)
    clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
    region_grouping = [
         [] for _ in range(n_regions)
    ]
    order = np.random.permutation(n_centers)
    for ridx, ord in enumerate(order):
        region_grouping[ridx % n_regions].append(ord)
    regions = {
        i: np.any(
            np.stack([clusters == j for j in region_grouping[i]], axis=2),
            axis=2
        ).astype(np.float32) for i in range(len(region_grouping))
    }
    return regions, centers


class Bayesian(nn.Module):
    '''
    p(x) ~ N(0, S)
    p(y | x) ~ N(Fx, Sigma) where Fx is FGL(x)... can be represented as a linear transform
    => p(x|y) ~ N(F.T @ inv(Sigma) @ y, inv(inv(S) + F.T @ inv(Sigma) @ F))

    Getting the F matrix:
        y = (A @ ((x @ v) * u)) @ w
        assume u is constant across channels for each position...
        then, this is the same as:
            y = A' @ (x @ v) @ w where A' is adjacency with values at non-zero positon (not binary)
             = A' @ x @ W
             = W @ A @ x for some W.
    '''
    def __init__(self, args, s=32, n=10000, n_regions=8, n_centers=16, y_uncertainity=0.1, prior_cov_mode='eye', likelihood_cov_mode='eye', try_loading=True):
        super().__init__()
        dump_to_file = ""

        if try_loading:  # Annoying code that I use to cache dataset.
            probable_file_dir = "/data/brain_/clevr/outputs/dataset_objects"
            probable_file_name = "bayesian_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(args.dset_seed, s, n, n_regions, n_centers, y_uncertainity, prior_cov_mode, likelihood_cov_mode)
            file_path = os.path.join(probable_file_dir, probable_file_name)
            os.makedirs(probable_file_dir, exist_ok=True)
            if os.path.isfile(file_path):
                print('loading from {}'.format(file_path))
                with open(file_path, 'rb') as f:
                    obj = pickle.load(f)
                self.__dict__.update(obj.__dict__)
            else:
                dump_to_file = file_path

        if dump_to_file != "":
            tic = time.time()
            self.n = n
            self.s = s
            self.regions, self.centers = exclusive_regions(args.dset_seed, s=s, n_regions=n_regions, n_centers=n_centers)
            self.n_regions = n_regions
            self.n_centers = n_centers
            self.xdim = (s * s, 1)
            self.hdim = (n_regions, 2)
            self.n_classes = np.prod(self.hdim)
            assert(self.n % self.n_classes == 0)
            self.ydim = (self.n_classes, )
            self.y_uncertainity = y_uncertainity

            # Compute A, F
            self.A = get_Ahx(self.s, self.regions)  # n_regions, s^2 matrix.
            v = np.random.randn(self.hdim[1], self.xdim[1])
            self.V = scipy.linalg.block_diag(*([v for _ in range(self.A.shape[0])]))
            self.W = np.random.randn(self.n_classes, np.prod(self.hdim))
            # self.W = np.eye(self.n_classes)
            self.F = np.einsum('ij,jk,kl->il', self.W, self.V, self.A)

            if prior_cov_mode == 'eye':
                self.S = np.eye(self.s * self.s, self.s * self.s)
            elif prior_cov_mode == 'infer':
                self.S = np.matmul(self.A.T, self.A)
            self.invS = np.linalg.pinv(self.S)
            if likelihood_cov_mode == 'eye':
                self.Sigma = np.eye(self.n_classes, self.n_classes)
            elif likelihood_cov_mode == 'infer':
                self.Sigma = np.matmul(self.W, self.W.T)
            self.invSigma = np.linalg.pinv(self.Sigma)

            self.posterior_cov = np.linalg.pinv(self.invS + (self.F.T @ self.invSigma @ self.F))
            self.posterior_mean_factor = self.F.T @ self.invSigma
            print('factors generation took {}s'.format(time.time() - tic))

            tic = time.time()
            self.X, self.Y = self.generate(args.dset_seed)
            print("generation of dataset took {}s".format(time.time() - tic))
            with open(dump_to_file, 'wb') as f:
                pickle.dump(self, f)

    def generate(self, seed):
        def _sample(seed, n, y_uncertainity, posterior_mean_factor, posterior_cov):
            np.random.seed(seed)
            nc = self.ydim[0]
            ys = np.arange(nc) # np.random.randint(0, nc, (n,))
            ys_vec = np.zeros((nc, nc)) + (y_uncertainity / (nc - 1))
            ys_vec[np.arange(nc), ys] = 1 - y_uncertainity
            xs = np.stack([np.random.multivariate_normal(posterior_mean_factor @ ys_vec[i], posterior_cov, size=(n // nc)) for i in range(nc)])
            xs = np.reshape(xs, (n, self.s, self.s))
            ys = np.reshape(np.stack([ys for _ in range(n // nc)], axis=-1), (n,))
            ys_vec = np.zeros((n, nc)) + (y_uncertainity / (nc - 1))
            ys_vec[np.arange(n), ys] = 1 - y_uncertainity
            yys = np.exp(ys_vec - ys_vec.max(1, keepdims=True))
            yys = yys / yys.sum(1, keepdims=True)
            confidence = yys[np.arange(n), np.argsort(yys, axis=1)[:, 0]] - yys[np.arange(n), np.argsort(yys, axis=1)[:, 1]]
            return xs.astype(np.float32), ys, confidence

        njobs = 1
        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.y_uncertainity,
                self.posterior_mean_factor,
                self.posterior_cov,
            ) for i in range(njobs)
        )
        X, Y, self.conf = list(zip(*samples))
        xs = np.concatenate(X)
        xs = (xs - xs.mean(0)) / xs.std(0)
        ys = np.concatenate(Y)
        self.conf = np.concatenate(self.conf)
        return xs.astype(np.float32), ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class OrderRegionGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, n_regions=8, n_centers=32, x_noise=1e-5):
        super().__init__()
        self.n = n
        self.s = s
        self.regions, self.centers = self.init_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)
        self.n_classes = self.n_regions = len(self.regions)
        self.x_noise = x_noise
        self.colors = np.random.randn(self.n_regions)
        # self.Wyh = (sp.rand(n_regions, n_regions, density=0.5, random_state=args.dset_seed).toarray()).astype(np.float32)
        self.Wyh = np.random.randn(n_regions, n_regions)

        tic = time.time()
        self.X, self.Y = self.generate_x(args.dset_seed)
        print("generation of dataset took {}s".format(time.time() - tic))

    def generate_x(self, seed):
        def _sample(seed, n, Wyh, x_noise):
            np.random.seed(seed)
            xs = []
            ys = []
            for i in range(n):
                order = np.random.permutation(self.n_regions)
                colors = self.colors + x_noise * np.random.rand(*(order.shape))
                xs.append(sum([colors[i] * self.regions[order[i]] for i in range(self.n_regions)], 0))
                ys.append(np.argmax(np.matmul(Wyh, colors[order])))
            return np.stack(xs, axis=0), np.stack(ys, axis=0), np.zeros(len(ys))
        njobs = 1
        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.Wyh,
                self.x_noise,
            ) for i in range(njobs)
        )
        X, Y, self.conf = list(zip(*samples))
        xs = np.concatenate(X)
        xs = (xs - xs.mean(0)) / xs.std(0)
        ys = np.concatenate(Y)
        self.conf = np.concatenate(self.conf)
        return xs.astype(np.float32), ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def init_regions(self, seed, n_regions=16, n_centers=8):
        np.random.seed(seed)
        random.seed(seed)
        s = self.s
        regions = np.zeros((s, s)).astype(np.int64)
        xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
        xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
        centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
        distances = scipy.spatial.distance.cdist(xypos, centers)
        clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
        region_grouping = [
             [] for _ in range(n_regions)
        ]
        for cidx in range(n_centers):
            region_grouping[np.random.randint(0, n_regions)].append(cidx)
        for ridx in range(len(region_grouping)):
            region_grouping[ridx].extend(random.sample(list(range(n_centers)), 2 * n_centers // n_regions))
            region_grouping[ridx] = list(set(region_grouping[ridx]))

        regions = {
            i: np.any(
                np.stack([clusters == j for j in region_grouping[i]], axis=2),
                axis=2
            ).astype(np.float32) for i in range(len(region_grouping))
        }
        return regions, centers

class RandomClusterGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, ydim=(8,), n_regions=32, n_centers=512, h_noise=0.1, xy_noise=5e-5, lmbda=0.0):
        super().__init__()
        self.n = n
        self.s = s
        self.xdim = xdim = (s * s, 1)
        self.hdim = hdim = (n_regions, 1)
        self.ydim = ydim
        self.h_noise = h_noise
        self.xy_noise = xy_noise
        # self.Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=args.dset_seed).toarray()  # Positive
        self.regions, self.centers = self.init_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)
        self.n_regions = len(self.regions)

        self.Ahx = self.get_Ahx(self.regions)
        self.Whx = np.random.randn(xdim[1], hdim[1])
        # self.Wyh = np.random.randn(ydim[0], np.prod(hdim))
        self.Wyh = (sp.rand(ydim[0], np.prod(hdim), density=0.5, random_state=args.dset_seed).toarray()).astype(np.float32)
        self.Wyh *= np.random.randn(*(self.Wyh.shape))
        self.n_classes = ydim[0]

        self.mu_yh = np.random.randn(ydim[0], hdim[0])  # 0 centerd
        self.cov_yh = []
        for i in range(ydim[0]):
            std_yh = np.random.randn(hdim[0], hdim[0])
            self.cov_yh.append(np.dot(std_yh.T, std_yh) + lmbda * np.eye(hdim[0]))
        self.cov_yh = np.stack(self.cov_yh)
        tic = time.time()
        self.X, self.Y = self.generate_x(args.dset_seed)
        print("generation of dataset took {}s".format(time.time() - tic))

    def get_Ahx(self, regions):
        Ahx = np.zeros((len(regions), self.s * self.s))
        for ridx, cis in regions.items():
            Ahx[ridx, np.reshape(cis, (-1,)) > 0] = 1.0
        return Ahx

    def init_regions(self, seed, n_regions=16, n_centers=8):
        np.random.seed(seed)
        random.seed(seed)
        s = self.s
        regions = np.zeros((s, s)).astype(np.int64)
        xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
        xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
        centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
        distances = scipy.spatial.distance.cdist(xypos, centers)
        clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
        region_grouping = [
             [] for _ in range(n_regions)
        ]
        for cidx in range(n_centers):
            region_grouping[np.random.randint(0, n_regions)].append(cidx)
        for ridx in range(len(region_grouping)):
            region_grouping[ridx].extend(random.sample(list(range(n_centers)), 2 * n_centers // n_regions))
            region_grouping[ridx] = list(set(region_grouping[ridx]))

        regions = {
            i: np.any(
                np.stack([clusters == j for j in region_grouping[i]], axis=2),
                axis=2
            ).astype(np.float32) for i in range(len(region_grouping))
        }
        return regions, centers

    def generate_x(self, seed, njobs=1):
        # def _sample(seed, n, mu, cov, Ahx, Whx, Wyh, h_noise, xy_noise):
        #     # x = Linear(h) and h = GMM(y)
        #     np.random.seed(seed)
        #     ys = np.random.randint(0, self.ydim[0], (n,))
        #     hs = np.stack([np.random.multivariate_normal(mu[ys[i]], cov[ys[i]]) for i in range(n)])  # 0 centered
        #     hs += h_noise * np.random.randn(*(hs.shape))  # Still 0 centered.
        #     xs = np.stack([np.matmul(np.matmul(Ahx.T, np.reshape(hs[i], (-1, 1))), Whx) for i in range(n)])
        #     xs += xy_noise * np.random.randn(*(xs.shape))
        #     return xs.astype(np.float32), ys, np.zeros(ys.shape)

        def _sample(seed, n, mu, cov, Ahx, Whx, Wyh, h_noise, xy_noise):
            # Same as y-> x but using matrix inverse instead of
            np.random.seed(seed)
            ys = np.random.randint(0, self.ydim[0], (n,))
            ys_vec = np.zeros((n, self.ydim[0]))
            ys_vec[np.arange(n), ys] = 1
            hs = np.matmul(np.linalg.pinv(Wyh), ys_vec)
            # hs = 2 * (hs - hs.min()) / (hs.max() - hs.min()) - 1
            hs += h_noise * np.random.randn(*(hs.shape))
            pinv_Ahx = np.linalg.pinv(Ahx),
            xs = np.stack([
                np.matmul(
                    np.matmul(
                        pinv_Ahx,
                        np.reshape(hs[i], (-1, 1))),
                    Whx
                ) for i in range(n)
            ])
            # 2 * (xs - xs.min()) / (xs.max() - xs.min()) - 1
            # xs += xy_noise * np.random.randn(*(xs.shape))
            return xs, ys, np.zeros(ys.shape)

        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.mu_yh,
                self.cov_yh,
                self.Ahx,
                self.Whx,
                self.Wyh,
                self.h_noise,
                self.xy_noise,
            ) for i in range(njobs)
        )
        X, Y, self.conf = list(zip(*samples))
        xs = np.concatenate(X)
        xs = (xs - xs.mean(0)) / xs.std(0)
        ys = np.concatenate(Y)
        self.conf = np.concatenate(self.conf)
        return xs.astype(np.float32), ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return np.reshape(x, (self.s, self.s)), y


class IIDHClusterGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, n_regions=20, n_centers=32, h_noise=0.5, x_noise=1.0, y_noise=0.25, lmbda=0.0):
        super().__init__()
        self.n = n
        self.s = s
        self.xdim = xdim = (s * s, 1)
        self.hdim = hdim = (n_regions, 1)
        self.ydim = ydim = (np.prod(hdim),)
        self.h_noise = h_noise
        self.x_noise = x_noise
        self.y_noise = y_noise
        # self.Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=args.dset_seed).toarray()  # Positive
        self.regions, self.centers = self.init_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)
        self.n_regions = len(self.regions)

        self.Ahx = self.get_Ahx(self.regions)
        self.Whx = np.random.randn(xdim[1], hdim[1])  # Shouldn't be used..
        self.Wyh = np.random.randn(ydim[0], np.prod(hdim))
        # self.Wyh = (sp.rand(ydim[0], np.prod(hdim), density=0.5, random_state=args.dset_seed).toarray()).astype(np.float32)
        # self.Wyh *= np.random.randn(*(self.Wyh.shape))
        self.n_classes = ydim[0]

        self.mu_yh = np.random.randn(ydim[0], hdim[0])  # 0 centerd
        self.cov_yh = []
        for i in range(ydim[0]):
            std_yh = np.random.randn(hdim[0], hdim[0])
            self.cov_yh.append(np.dot(std_yh.T, std_yh) + lmbda * np.eye(hdim[0]))
        self.cov_yh = np.stack(self.cov_yh)
        tic = time.time()
        self.X, self.Y = self.generate_x(args.dset_seed)
        print("generation of dataset took {}s".format(time.time() - tic))

    def get_Ahx(self, regions):
        Ahx = np.zeros((len(regions), self.s * self.s))
        for ridx, cis in regions.items():
            Ahx[ridx, np.reshape(cis, (-1,)) > 0] = 1.0
        return Ahx

    def init_regions(self, seed, n_regions=8, n_centers=16):
        np.random.seed(seed)
        random.seed(seed)
        s = self.s
        regions = np.zeros((s, s)).astype(np.int64)
        xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
        xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
        centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
        distances = scipy.spatial.distance.cdist(xypos, centers)
        clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
        region_grouping = [
             [] for _ in range(n_regions)
        ]
        order = np.random.permutation(n_centers)
        for ridx, ord in enumerate(order):
            region_grouping[ridx % n_regions].append(ord)
        # for cidx in range(n_centers):
        #     region_grouping[np.random.randint(0, n_regions)].append(cidx)
        # for ridx in range(len(region_grouping)):
        #     region_grouping[ridx].extend(random.sample(list(range(n_centers)), 2 * n_centers // n_regions))
        #     region_grouping[ridx] = list(set(region_grouping[ridx]))

        regions = {
            i: np.any(
                np.stack([clusters == j for j in region_grouping[i]], axis=2),
                axis=2
            ).astype(np.float32) for i in range(len(region_grouping))
        }
        return regions, centers

    def generate_x(self, seed, njobs=1):
        def _sample(seed, n, mu, cov, Ahx, Whx, Wyh, h_noise, x_noise, y_noise):
            # x ~ N(0, 1), y = FGL(x).. good at 50k points, noise=0.1 for each, 32 regions, 512 clusters
            np.random.seed(seed)
            # hs = (-1 + 2 * np.random.rand(n, *(self.hdim))).astype(np.float32)
            # temp += h_noise * np.random.randn(*temp.shape)
            ys = np.random.randint(0, Wyh.shape[0], size=(n,))
            ys_vec = np.zeros((n, self.ydim[0]))
            ys_vec[np.arange(n), ys] = 1
            hs = np.reshape(np.einsum('yh,ny->nh', Wyh, ys_vec), (n, *(self.hdim))).astype(np.float32)
            hs += h_noise * np.random.randn(n, *(self.hdim)).astype(np.float32)
            temp = np.einsum('nhc,dc->nhd', hs, Whx)
            pinvAhx = np.linalg.pinv(Ahx)
            pinvAhx /= pinvAhx.max()
            xs = np.einsum('xh,nhd->nxd', pinvAhx, temp)
            xs += x_noise * np.random.randn(*(xs.shape))
            xs = np.reshape(xs, (n, self.xdim[0]))
            ys = np.einsum('nh,yh->ny', np.reshape(hs, (n, np.prod(self.hdim))), Wyh)
            ys = (ys - ys.mean(0)) / (ys.std(0))
            ys += y_noise * np.random.randn(*(ys.shape))
            yys = np.exp(ys - ys.max(1, keepdims=True))
            yys = yys / yys.sum(1, keepdims=True)
            confidence = yys[np.arange(n), np.argsort(yys, axis=1)[:, 0]] - yys[np.arange(n), np.argsort(yys, axis=1)[:, 1]]
            ys = np.argmax(yys, axis=1)
            return xs.astype(np.float32), ys, confidence

        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.mu_yh,
                self.cov_yh,
                self.Ahx,
                self.Whx,
                self.Wyh,
                self.h_noise,
                self.x_noise,
                self.y_noise,
            ) for i in range(njobs)
        )
        X, Y, self.conf = list(zip(*samples))
        xs = np.concatenate(X)
        xs = (xs - xs.mean(0)) / xs.std(0)
        ys = np.concatenate(Y)
        self.conf = np.concatenate(self.conf)
        return xs.astype(np.float32), ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return np.reshape(x, (self.s, self.s)), y


class IIDXClusterGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, ydim=(8,), n_regions=32, n_centers=512, h_noise=0.1, xy_noise=0.1, lmbda=0.0):
        super().__init__()
        self.n = n
        self.s = s
        self.xdim = xdim = (s * s, 1)
        self.hdim = hdim = (n_regions, 1)
        self.ydim = ydim
        self.h_noise = h_noise
        self.xy_noise = xy_noise
        # self.Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=args.dset_seed).toarray()  # Positive
        self.regions, self.centers = exclusive_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)  # self.init_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)
        self.n_regions = len(self.regions)

        self.Ahx = self.get_Ahx(self.regions)
        self.Whx = np.random.randn(xdim[1], hdim[1])
        self.Wyh = np.random.randn(ydim[0], np.prod(hdim))
        # self.Wyh = (sp.rand(ydim[0], np.prod(hdim), density=0.5, random_state=args.dset_seed).toarray()).astype(np.float32)
        # self.Wyh *= np.random.randn(*(self.Wyh.shape))
        self.n_classes = ydim[0]

        self.mu_yh = np.random.randn(ydim[0], hdim[0])  # 0 centerd
        self.cov_yh = []
        for i in range(ydim[0]):
            std_yh = np.random.randn(hdim[0], hdim[0])
            self.cov_yh.append(np.dot(std_yh.T, std_yh) + lmbda * np.eye(hdim[0]))
        self.cov_yh = np.stack(self.cov_yh)
        tic = time.time()
        self.X, self.Y = self.generate_x(args.dset_seed)
        print("generation of dataset took {}s".format(time.time() - tic))

    def get_Ahx(self, regions):
        Ahx = np.zeros((len(regions), self.s * self.s))
        for ridx, cis in regions.items():
            Ahx[ridx, np.reshape(cis, (-1,)) > 0] = 1.0
        return Ahx

    def init_regions(self, seed, n_regions=16, n_centers=8):
        np.random.seed(seed)
        random.seed(seed)
        s = self.s
        regions = np.zeros((s, s)).astype(np.int64)
        xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
        xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
        centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
        distances = scipy.spatial.distance.cdist(xypos, centers)
        clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
        region_grouping = [
             [] for _ in range(n_regions)
        ]
        for cidx in range(n_centers):
            region_grouping[np.random.randint(0, n_regions)].append(cidx)
        for ridx in range(len(region_grouping)):
            region_grouping[ridx].extend(random.sample(list(range(n_centers)), 2 * n_centers // n_regions))
            region_grouping[ridx] = list(set(region_grouping[ridx]))

        regions = {
            i: np.any(
                np.stack([clusters == j for j in region_grouping[i]], axis=2),
                axis=2
            ).astype(np.float32) for i in range(len(region_grouping))
        }
        return regions, centers

    def generate_x(self, seed, njobs=1):
        def _sample(seed, n, mu, cov, Ahx, Whx, Wyh, h_noise, xy_noise):
            # x ~ N(0, 1), y = FGL(x).. good at 50k points, noise=0.1 for each, 32 regions, 512 clusters
            np.random.seed(seed)
            xs = np.random.randn(n, *(self.xdim)).astype(np.float32)
            hs = np.stack([np.matmul(np.matmul(Ahx, xs[i]), Whx) for i in range(n)])
            # hs += h_noise * np.random.randn(*(hs.shape))  # Still 0 centered.
            ys = np.stack([np.matmul(Wyh, np.reshape(hs[i], (self.hdim[0],))) for i in range(n)])
            ys += xy_noise * np.random.randn(*(ys.shape))
            yys = np.exp(ys - ys.max(1, keepdims=True))
            yys = yys / yys.sum(1, keepdims=True)
            confidence = yys[np.arange(n), np.argsort(yys, axis=1)[:, -1]]#  - yys[np.arange(n), np.argsort(yys, axis=1)[:, -2]]
            ys = np.argmax(ys, axis=1)
            return xs.astype(np.float32), ys, confidence

        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.mu_yh,
                self.cov_yh,
                self.Ahx,
                self.Whx,
                self.Wyh,
                self.h_noise,
                self.xy_noise,
            ) for i in range(njobs)
        )
        X, Y, self.conf = list(zip(*samples))
        xs = np.concatenate(X)
        xs = (xs - xs.mean(0)) / xs.std(0)
        ys = np.concatenate(Y)
        self.conf = np.concatenate(self.conf)
        return xs.astype(np.float32), ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return np.reshape(x, (self.s, self.s)), y


class RandomGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, hdim=(32, 1), ydim=(8,), hx_density=0.01, h_noise=0.01, y_noise=0.01):
        super().__init__()
        self.n = n
        self.s = s
        self.xdim = xdim = (s * s, 1)
        self.hdim = hdim
        self.ydim = ydim
        self.h_noise = h_noise
        self.y_noise = y_noise
        self.Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=args.dset_seed).toarray()  # Positive
        self.Whx = np.random.randn(xdim[1], hdim[1])
        self.Wyh = np.random.randn(ydim[0], np.prod(hdim))
        self.regions = self.init_regions(self.Ahx)
        self.n_regions = len(self.regions)
        self.n_classes = ydim[0]

        self.X, self.Y = self.generate_x(args.dset_seed)

    def init_regions(self, Ahx):
        regions = [[] for _ in range(Ahx.shape[0])]
        rs, cs = np.where(Ahx)
        for i in range(rs.shape[0]):
            regions[rs[i]].append(cs[i])
        return regions

    def generate_x(self, seed, njobs=1):
        def _sample(seed, n, Ahx, Whx, Wyh, h_noise, y_noise):
            np.random.seed(seed)
            xs = np.random.randn(n, *(self.xdim)).astype(np.float32)
            hs = np.stack([np.matmul(np.matmul(Ahx, xs[i]), Whx) for i in range(n)])
            hs += h_noise * np.random.randn(*(hs.shape))  # Still 0 centered.
            ys = np.stack([np.matmul(Wyh, np.reshape(hs[i], (self.hdim[0],))) for i in range(n)])
            ys += y_noise * np.random.randn(*(ys.shape))
            ys = np.argmax(ys, axis=1)
            return xs, ys

        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                seed,
                ceil(self.n / njobs),
                self.Ahx,
                self.Whx,
                self.Wyh,
                self.h_noise,
                self.y_noise,
            ) for i in range(njobs)
        )
        X, Y = list(zip(*samples))
        xs = np.concatenate(X)
        ys = np.concatenate(Y)
        return xs, ys

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return np.reshape(x, (self.s, self.s)), y


class ClusterGrid(nn.Module):
    def __init__(self, args, s=128, n=50000, n_regions=4, n_centers=1000):
        super().__init__()
        self.n = n
        self.s = s
        self.pattern_generators = [
            # lambda: self.make_random_pattern(),
            # lambda: self.make_radial_pattern(),
            lambda: self.make_horizontal_linear_pattern(),
            lambda: self.make_vertical_linear_pattern(),
        ]
        self.n_pattern_types = len(self.pattern_generators)  # random, radial, horizontal linear, vertical linear
        self.regions, self.centers = self.init_regions(args.dset_seed, n_regions=n_regions, n_centers=n_centers)
        self.n_regions = len(self.regions)
        self.n_classes = self.n_regions * self.n_pattern_types
        print("n_classes={}".format(self.n_classes))

        # X noise generation parameters
        self.general_noise = 0.5  # Dampened noise.
        self.wedge_noise = 1.0
        self.pattern_alpha_cut = 1.0

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

    def init_regions(self, seed, n_regions=16, n_centers=8):
        np.random.seed(seed)
        random.seed(seed)
        s = self.s
        regions = np.zeros((s, s)).astype(np.int64)
        xpos, ypos = np.meshgrid(list(range(s)), list(range(s)), indexing='ij')
        xypos = np.reshape(np.stack([xpos, ypos], axis=-1), (s*s, 2))
        centers = np.random.randint(low=0, high=s, size=(n_centers, 2))
        distances = scipy.spatial.distance.cdist(xypos, centers)
        clusters = np.reshape(np.argmin(distances, axis=1), (s, s))
        # region_grouping = [[] for _ in range(n_regions)]
        # for cidx in range(n_centers):
        #     region_grouping[cidx % n_regions].append(cidx)
        region_grouping = [
             [] for _ in range(n_regions)
        ]
        for cidx in range(n_centers):
            region_grouping[np.random.randint(0, n_regions)].append(cidx)
        for ridx in range(len(region_grouping)):
            region_grouping[ridx].extend(random.sample(list(range(n_centers)), 2 * n_centers // n_regions))
            region_grouping[ridx] = list(set(region_grouping[ridx]))

        regions = {
            i: np.any(
                np.stack([clusters == j for j in region_grouping[i]], axis=2),
                axis=2
            ).astype(np.float32) for i in range(len(region_grouping))
        }
        return regions, centers

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
