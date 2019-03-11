import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.sampler import SubsetRandomSampler
from clevr import (
    notso,
    somewhat,
    wedges,
    grid,
)

datasets = {
    'notso': lambda args: notso.NotSoClevr(),
    'bignotso': lambda args: notso.NotSoClevr(s=128, l=15),
    'somewhat': lambda args: somewhat.SomewhatClevr(),
    'wedges10000': lambda args: wedges.Wedges(n=10000),
    'complexwedges10000': lambda args: wedges.ComplexWedges(args, n=10000),
    'complexwedges50000': lambda args: wedges.ComplexWedges(args, n=50000),
    'gradientwedges50000': lambda args: wedges.GradientWedges(args, n=50000),
    'clustergrid50000': lambda args: grid.ClusterGrid(args, n=50000),
    'randomgrid50000': lambda args: grid.RandomGrid(args, n=50000),
    'randomclustergrid10000': lambda args: grid.RandomClusterGrid(args, n=10000),
    'randomclustergrid50000': lambda args: grid.RandomClusterGrid(args, n=50000),
    'iidxclustergrid50000': lambda args: grid.IIDXClusterGrid(args, n=50000),  # GOOD!
    'iidhclustergrid50000': lambda args: grid.IIDHClusterGrid(args, n=50000),
    'orderregiongrid50000': lambda args: grid.OrderRegionGrid(args, n=50000),
    'bayesian16000': lambda args: grid.Bayesian(args, n=16000, n_regions=8, n_centers=10, prior_cov_mode='eye', likelihood_cov_mode='eye'),
    'blockybayesian16000': lambda args: grid.Bayesian(args, n=16000, n_regions=8, n_centers=10, prior_cov_mode='infer', likelihood_cov_mode='infer'),
}


def get_splits(args):
    splits = []
    meta = {}
    dataset = datasets[args.name](args)
    meta['n_classes'] = dataset.n_classes
    meta['s'] = dataset.s
    for attr_name in ['r', 'regions']:
        if hasattr(dataset, attr_name):
            meta[attr_name] = getattr(dataset, attr_name)

    if args.debug:
        dataset.n = 40
    dataset_size = len(dataset)
    np.random.seed(args.dset_seed)
    for splitidx in range(args.outer_folds):
        indices = np.random.permutation(dataset_size)
        testing_end = int(dataset_size * args.outer_frac)
        training_end = testing_end + int(dataset_size * args.training_frac)
        test_indices, train_indices = indices[:testing_end], indices[testing_end: training_end]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

        splits.append({
            'test': test_loader,
            'splits': [{'train': train_loader}]}
        )
    return splits, meta
