
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

# X ~ N(0, 1)
# h = AxW + eps
# y = Uh + eps
ydim = (8,)
hdim = (128, 1)
xdim = (784, 3)


class ToyDataset(TorchDataset):
    def __init__(self, npz, start=0, end=0):
        self.X = npz["X"][start: end, ...].astype(np.float32)
        self.Y = npz["Y"][start: end, ...].astype(np.int64)
        self.Ahx = npz["Ahx"]
        self.Whx = npz["Whx"]
        self.Wyh = npz["Wyh"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i].T, self.Y[i]


def load_train_test(location, train_frac):
    npz = np.load(location)
    n = npz["X"].shape[0]
    train_end = int(train_frac * n)
    trn = ToyDataset(npz, 0, train_end)
    tst = ToyDataset(npz, train_end, n)
    return trn, tst


def _sample(seed, n, Ahx, Whx, Wyh, hn, xn):
    np.random.seed(seed)

    xs = np.random.randn(n, *xdim).astype(np.float32)
    hs = np.stack([np.matmul(np.matmul(Ahx, xs[i]), Whx) for i in range(n)])
    hs += h_noise * np.random.randn(*(hs.shape))  # Still 0 centered.

    ys = np.stack([np.matmul(Wyh, np.reshape(hs[i], (hdim[0],))) for i in range(n)])
    ys += y_noise * np.random.randn(*(ys.shape))
    ys = np.argmax(ys, axis=1)
    return xs, ys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str, help="Where to save the generated data")
    parser.add_argument("-s", "--seed", type=int, default=1337, help="Seed for dataset generation")
    parser.add_argument("-n", "--n", type=int, default=10000, help="Number of datapoints to sample")
    parser.add_argument("-hn", "--h_noise", dest="h_noise", type=float, default=0.0001, help="Std dev of gauss noise added to h")
    parser.add_argument("-yn", "--y_noise", dest="y_noise", type=float, default=0.0001, help="Std dev of gauss noise added to y")
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
        y_noises = [0.05]
        for (
            n,
            seed,
            lmbda,
            hx_density,
            njobs,
            h_noise,
            y_noise,
        ) in itertools.product(
            ns,
            seeds,
            lmbdas,
            hx_densitys,
            njobss,
            h_noises,
            y_noises,
        ):
            location = os.path.join(args.location, "data_n{}_seed{}_hxd{}_hn{}_yn{}.npz".format(n, seed, hx_density, h_noise, y_noise))
            combination = {
                'n': n,
                'seed': seed,
                'lmbda': lmbda,
                'hx_density': hx_density,
                'njobs': njobs,
                'h_noise': h_noise,
                'y_noise': y_noise,
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
            'y_noise': args.y_noise,
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
        y_noise = combination['y_noise']
        location = combination['location']
        print("[{}/{}] Creating dataset at {}".format(dsetidx, len(combinations), location))

        # Create and dump dataset
        np.random.seed(seed)

        Ahx = sp.rand(hdim[0], xdim[0], density=hx_density, random_state=2 * seed).toarray()  # Positive
        Whx = np.random.randn(xdim[1], hdim[1])
        Wyh = np.random.randn(ydim[0], np.prod(hdim))
        print("Created parameters")

        start = time.time()
        # samples = _sample(seed, n, mu_yz, cov_yz, Azh, Wzh, Ahx, Whx, z_noise, h_noise, y_noise)
        samples = joblib.Parallel(njobs)(
            joblib.delayed(_sample)(
                i * seed,
                ceil(n / njobs),
                Ahx,
                Whx,
                Wyh,
                h_noise,
                y_noise,
            ) for i in range(njobs)
        )
        print("Sampled")
        X, Y = list(zip(*samples))
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        os.makedirs(os.path.dirname(location), exist_ok=True)
        print("Saving to {}".format(location))
        np.savez(location, X=X, Y=Y, Wyh=Wyh, Whx=Whx, Ahx=Ahx)
        print("{} took {:.2f}".format(location, time.time() - start))
        trn, tst = load_train_test(location, 0.8)
