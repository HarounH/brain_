"""
Toy dataset for Graph-like-MNIST

data generation process:
c -> z -> h -> x(mat)
c: 8 potential classes,
z: essentially an embedding for c. (128)
h: intermediate variable (256, 4)
x: (784, 3) 0 centered, 1 std dev for each of the 784 * 3 values.
z->h and h->x are both determined by sparse matrices which can be sampled.
"""

import argparse
import pdb
import os
import time
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
import scipy.sparse as sp
import joblib
from math import ceil
import itertools

xdim = (784, 3)  # 2352 numbers
hdim = (256, 4)
zdim = (128,)
ydim = (8,)


class ToyDataset(TorchDataset):
    def __init__(self, npz, start=0, end=0):
        self.X = npz["X"][start: end, ...].astype(np.float32)
        self.Y = npz["Y"][start: end, ...].astype(np.int64)
        self.Ahx = npz["Ahx"]
        self.Azh = npz["Azh"]
        self.Whx = npz["Whx"]
        self.Wzh = npz["Wzh"]
        self.mu_yz = npz["mu_yz"]
        self.cov_yz = npz["cov_yz"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def load_train_test(location, train_frac):
    npz = np.load(location)
    n = npz["X"].shape[0]
    train_end = int(train_frac * n)
    trn = ToyDataset(npz, 0, train_end)
    tst = ToyDataset(npz, train_end, n)
    return trn, tst


def _sample(seed, n, mu, cov, Azh, Wzh, Ahx, Whx, z_noise=1e-3, h_noise=1e-3, x_noise=1e-3):  # n (int): number of samples to get
    np.random.seed(seed)
    ys = np.random.randint(0, ydim[0], (n,))
    zs = np.stack([np.random.multivariate_normal(mu[ys[i]], cov[ys[i]]) for i in range(n)])  # 0 centered
    zs += z_noise * np.random.rand(*(zs.shape))  # Still 0 centered.

    hs = np.stack([np.matmul(np.reshape(np.matmul(zs[i], Azh), (hdim[0], 1)), Wzh) for i in range(n)])
    hs += h_noise * np.random.rand(*(hs.shape))

    xs = np.stack([np.matmul(np.matmul(Ahx.T, hs[i]), Whx) for i in range(n)])
    xs += x_noise * np.random.rand(*(xs.shape))
    return xs, ys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str, help="Where to save the generated data")
    parser.add_argument("-s", "--seed", type=int, default=1337, help="Seed for dataset generation")
    parser.add_argument("-n", "--n", type=int, default=10000, help="Number of datapoints to sample")
    parser.add_argument("-zn", "--z_noise", dest="z_noise", type=float, default=0.001, help="Std dev of gauss noise added to z")
    parser.add_argument("-hn", "--h_noise", dest="h_noise", type=float, default=0.001, help="Std dev of gauss noise added to h")
    parser.add_argument("-xn", "--x_noise", dest="x_noise", type=float, default=0.001, help="Std dev of gauss noise added to x")
    parser.add_argument("-zh", "--zh_density", dest="zh_density", type=float, default=0.01, help="Density for sparse matrix from z to h")
    parser.add_argument("-hx", "--hx_density", dest="hx_density", type=float, default=0.01, help="Density for sparse matrix from h to x")
    parser.add_argument("-nj", "--njobs", type=int, default=2, help="Number of jobs to use for joblib.")
    parser.add_argument("-l", "--lmbda", type=float, default=0.0, help="Lambda added to cov matrices to ensure positive semidefinite")
    args = parser.parse_args()

    if os.path.isdir(args.location):  # Generate a bunch of data.
        combinations = []
        ns = [10000]
        seeds = [1337, 7, 42, 666, 13, 8]
        lmbdas = [0.0]
        njobss = [4]
        zh_densitys = [0.01, 0.2, 0.4]
        hx_densitys = [0.01, 0.2, 0.4]
        z_noises = [0.05]
        h_noises = [0.05]
        x_noises = [0.05]
        for (
            n,
            seed,
            lmbda,
            zh_density,
            hx_density,
            njobs,
            z_noise,
            h_noise,
            x_noise,
        ) in itertools.product(
            ns,
            seeds,
            lmbdas,
            zh_densitys,
            hx_densitys,
            njobss,
            z_noises,
            h_noises,
            x_noises,
        ):
            location = os.path.join(args.location, "data_n{}_seed{}_zhd{}_hxd{}_zn{}_hn{}_xn{}.npz".format(n, seed, zh_density, hx_density, z_noise, h_noise, x_noise))
            combination = {
                'n': n,
                'seed': seed,
                'lmbda': lmbda,
                'zh_density': zh_density,
                'hx_density': hx_density,
                'njobs': njobs,
                'z_noise': z_noise,
                'h_noise': h_noise,
                'x_noise': x_noise,
                'location': location,
            }
            combinations.append(combination)
    else:
        combination = {
            'n': args.n,
            'seed': args.seed,
            'lmbda': args.lmbda,
            'zh_density': args.zh_density,
            'hx_density': args.hx_density,
            'njobs': args.njobs,
            'z_noise': args.z_noise,
            'h_noise': args.h_noise,
            'x_noise': args.x_noise,
            'location': args.location,
        }
        combinations = [combination]

    for dsetidx, combination in enumerate(combinations):
        n = combination['n']
        seed = combination['seed']
        lmbda = combination['lmbda']
        zh_density = combination['zh_density']
        hx_density = combination['hx_density']
        njobs = combination['njobs']
        z_noise = combination['z_noise']
        h_noise = combination['h_noise']
        x_noise = combination['x_noise']
        location = combination['location']
        print("[{}/{}] Creating dataset at {}".format(dsetidx, len(combinations), location))

        # Create and dump dataset
        np.random.seed(seed)

        mu_yz = np.random.rand(ydim[0], zdim[0])  # 0 centerd
        cov_yz = []
        for i in range(ydim[0]):
            std_yz = np.random.rand(zdim[0], zdim[0])
            cov_yz.append(np.dot(std_yz.T, std_yz) + lmbda * np.eye(zdim[0]))
        cov_yz = np.stack(cov_yz)
        Azh = sp.rand(zdim[0], hdim[0], density=zh_density, random_state=3 * seed).toarray()  # Positive
        Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=2 * seed).toarray()  # Positive
        Wzh = np.random.rand(1, hdim[1])
        Whx = np.random.rand(hdim[1], xdim[1])
        print("Created parameters")
        start = time.time()
        # samples = _sample(seed, n, mu_yz, cov_yz, Azh, Wzh, Ahx, Whx, z_noise, h_noise, x_noise)
        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                i * seed,
                ceil(n / njobs),
                mu_yz,
                cov_yz,
                Azh,
                Wzh,
                Ahx,
                Whx,
                z_noise,
                h_noise,
                x_noise,
            ) for i in range(njobs)
        )
        print("Sampled")
        X, Y = list(zip(*samples))
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        os.makedirs(os.path.dirname(location), exist_ok=True)
        print("Saving to {}".format(location))
        np.savez(location, X=X, Y=Y, mu_yz=mu_yz, cov_yz=cov_yz, Wzh=Wzh, Whx=Whx, Azh=Azh, Ahx=Ahx)
        print("{} took {:.2f}".format(location, time.time() - start))
        trn, tst = load_train_test(location, 0.8)
