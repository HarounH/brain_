import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.sampler import SubsetRandomSampler
from clevr import (
    notso,
    somewhat,
)

datasets = {
    'notso': notso.NotSoClevr,
    'bignotso': lambda: notso.NotSoClevr(s=128, l=15),
    'somewhat': somewhat.SomewhatClevr,
}


def get_splits(args):
    splits = []
    meta = {}
    dataset = datasets[args.name]()
    meta['n_classes'] = dataset.n_classes
    meta['s'] = dataset.s
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
