
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

# GMM y -> h
# Sparse linear h -> x
ydim = (8,)
hdim = (128, 1)
xdim = (784, 3)


class ToyDataset(TorchDataset):
    def __init__(self, npz, start=0, end=0):
        self.X = npz["X"][start: end, ...].astype(np.float32)
        self.Y = npz["Y"][start: end, ...].astype(np.int64)
        self.Ahx = npz["Ahx"]
        self.Whx = npz["Whx"]
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


def _sample(seed, n, mu, cov, Ahx, Whx, hn, xn):
    np.random.seed(seed)
    ys = np.random.randint(0, ydim[0], (n,))

    hs = np.stack([np.random.multivariate_normal(mu[ys[i]], cov[ys[i]]) for i in range(n)])  # 0 centered
    hs += h_noise * np.random.rand(*(hs.shape))  # Still 0 centered.

    xs = np.stack([np.matmul(np.matmul(Ahx.T, np.reshape(hs[i], (-1, 1))), Whx) for i in range(n)])
    xs += x_noise * np.random.rand(*(xs.shape))
    return xs, ys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str, help="Where to save the generated data")
    parser.add_argument("-s", "--seed", type=int, default=1337, help="Seed for dataset generation")
    parser.add_argument("-n", "--n", type=int, default=10000, help="Number of datapoints to sample")
    parser.add_argument("-hn", "--h_noise", dest="h_noise", type=float, default=0.0001, help="Std dev of gauss noise added to h")
    parser.add_argument("-xn", "--x_noise", dest="x_noise", type=float, default=0.0001, help="Std dev of gauss noise added to x")
    parser.add_argument("-hx", "--hx_density", dest="hx_density", type=float, default=0.0001, help="Density for sparse matrix from h to x")
    parser.add_argument("-nj", "--njobs", type=int, default=2, help="Number of jobs to use for joblib.")
    parser.add_argument("-l", "--lmbda", type=float, default=0.0, help="Lambda added to cov matrices to ensure positive semidefinite")
    args = parser.parse_args()


    if os.path.isdir(args.location):  # Generate a bunch of data.
        combinations = []
        ns = [10000]
        seeds = [1337, 42, 666]  # , 13, 8, 42, 53, 99, 10412398, 31235213, 13213, 526456, 12312321]
        lmbdas = [0.0]
        njobss = [4]
        hx_densitys = [0.01, 0.05, 0.2, 0.5]
        h_noises = [0.05]
        x_noises = [0.05]
        for (
            n,
            seed,
            lmbda,
            hx_density,
            njobs,
            h_noise,
            x_noise,
        ) in itertools.product(
            ns,
            seeds,
            lmbdas,
            hx_densitys,
            njobss,
            h_noises,
            x_noises,
        ):
            location = os.path.join(args.location, "data_n{}_seed{}_hxd{}_hn{}_xn{}.npz".format(n, seed, hx_density, h_noise, x_noise))
            combination = {
                'n': n,
                'seed': seed,
                'lmbda': lmbda,
                'hx_density': hx_density,
                'njobs': njobs,
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
            'hx_density': args.hx_density,
            'njobs': args.njobs,
            'h_noise': args.h_noise,
            'x_noise': args.x_noise,
            'location': args.location,
        }
        combinations = [combination]


    for dsetidx, combination in enumerate(combinations):
        n = combination['n']
        seed = combination['seed']
        lmbda = combination['lmbda']
        hx_density = combination['hx_density']
        njobs = combination['njobs']
        h_noise = combination['h_noise']
        x_noise = combination['x_noise']
        location = combination['location']
        print("[{}/{}] Creating dataset at {}".format(dsetidx, len(combinations), location))

        # Create and dump dataset
        np.random.seed(seed)

        mu_yz = np.random.rand(ydim[0], hdim[0])  # 0 centerd
        cov_yz = []
        for i in range(ydim[0]):
            std_yz = np.random.rand(hdim[0], hdim[0])
            cov_yz.append(np.dot(std_yz.T, std_yz) + lmbda * np.eye(hdim[0]))
        cov_yz = np.stack(cov_yz)

        Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=2 * seed).toarray()  # Positive
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
                Ahx,
                Whx,
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
        np.savez(location, X=X, Y=Y, mu_yz=mu_yz, cov_yz=cov_yz, Whx=Whx, Ahx=Ahx)
        print("{} took {:.2f}".format(location, time.time() - start))
        trn, tst = load_train_test(location, 0.8)
