"""
Perform training/testing by splitting the dataset multiple times.
No validation set is maintained.
Instead, we keep track of best (on test) and latest models
"""


import os
import json
import argparse
import numpy as np
import scipy
import random
from random import shuffle
import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
import multiprocessing
# pytorch imports
import torch
from torch import autograd
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transfer.modules import (
    classifiers,
)
import utils.utils as utils
from data import (
    dataset,
    constants
)
import sklearn.metrics as sk_metrics

DEBUG_N = 40

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("studies", type=str, choices=constants.nv_ids.keys(), nargs='+', help="dataset to use")  # noqa
    parser.add_argument("-ok", "--outer_folds", type=int, metavar='<int>', default=10, help="Number of outer folds")  # noqa
    parser.add_argument("-df", "--outer_frac", type=float, metavar='<int>', default=0.3, help="Fraction of data to use as test in each outer fold")  # noqa
    parser.add_argument("-ds", "--dset_seed", type=int, metavar='<int>', default=1337, help="Seed used for dataset")  # noqa

    # Data parameters... don't mess with this.
    parser.add_argument("--downsampled", default=False, action="store_true", help="Use downsampled (to BASC template) data")
    parser.add_argument("--not_lazy", default=False, action="store_true", help="If provided, all data is loaded right away instead of being loaded on the fly")
    parser.add_argument("-nrm", "--normalization", dest="normalization", choices=['none', 'both', '0c', '11'], default="none", help="What kind of normalization to use")
    parser.add_argument("--dbg", dest="debug", default=False, action="store_true", help="if true, then set to 40 datapoints in each dataset.")  # noqa

    parser.add_argument("--base_output", dest="base_output", default="transfer/outputs/multi_run/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa

    # System parameters
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    # Choose the learning parameters
    parser.add_argument("-es", "--epoch_size", dest="epoch_size", type=int, default=250, help="Number of batches in each batch")
    parser.add_argument("-maxb", "--max_batches", dest="max_batches", type=int, default=1000, help="Maximum number of batches to run for")
    parser.add_argument("-minb", "--min_batches", dest="min_batches", type=int, default=100, help="Minimum number of batches to run for")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Choose the model
    parser.add_argument('-ct', "--classifier_type", type=str, default='fc', help="What classifier version to use",)  # choices=classifiers.versions.keys(),)
    parser.add_argument('-drp', '--dropout', default=False, action="store_true", help="if true, then use dropout in the models.")  # noqa

    args = parser.parse_args()

    args.num_workers = multiprocessing.cpu_count() // 3
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.base_output = os.path.join(args.base_output, args.classifier_type)
    if args.debug:
        args.run_code = "debug"
    os.makedirs(args.base_output, exist_ok=True)

    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    args.base_output = os.path.join(args.base_output, args.run_code)
    os.makedirs(args.base_output, exist_ok=True)
    print("Using run_code: {}".format(args.run_code))

    return args


def evaluate(args, model, loaders_dict):
    metrics_by_study = {}
    model = model.eval()
    with torch.no_grad():
        for study, loader in loaders_dict.items():
            tic = time.time()
            nclasses = len(args.meta['si2ci'][args.meta['s2i'][study]])
            ctrues = []
            cpreds = []
            metrics = {
                'loss': 0.0,
            }
            for bidx, (x, _, _, cvec) in enumerate(loader):
                N = x.shape[0]
                if args.cuda:
                    # If dataparallel, then nn.DataParallel will automatically send stuff to the correct device
                    x = x.cuda()
                    cvec = cvec.cuda()

                study_vec = torch.tensor([args.meta['s2i'][study] for _ in range(N)], device=x.device).int()
                offset = min(args.meta['si2ci'][args.meta['s2i'][study]])
                cvec -= offset

                cpred = model(study_vec, x)
                metrics['loss'] += N * F.cross_entropy(cpred, cvec).item()
                ctrues.extend(cvec.cpu().tolist())
                cpreds.extend(torch.argmax(cpred.detach(), dim=1).cpu().tolist())

            metrics['loss'] /= len(loader)
            metrics['cm'] = sk_metrics.confusion_matrix(ctrues, cpreds, labels=list(range(nclasses)))
            metrics['accuracy'] = sk_metrics.accuracy_score(ctrues, cpreds)
            metrics['precision'], metrics['recall'], metrics['f1'], metrics['support'] = sk_metrics.precision_recall_fscore_support(ctrues, cpreds, labels=list(range(nclasses)))
            metrics['time'] = time.time() - tic
            metrics_by_study[study] = metrics
    model = model.train()
    return metrics_by_study


def run_single_split(args, split_by_study, epoch_size=10, output_dir="."):
    train_metrics_per_epoch = []
    test_metrics_per_epoch = []

    test_dset_by_study = {study: split['test'] for study, split in split_by_study.items()}
    train_dset_by_study = {study: split['splits'][0]['train'] for study, split in split_by_study.items()}
    if args.debug:
        for study in test_dset_by_study.keys():
            test_dset_by_study[study].n = DEBUG_N
            train_dset_by_study[study].n = DEBUG_N

    if args.not_lazy:
        for study in train_dset_by_study.keys():
            if study in ["archi", "la5c"]:
                print("Starting {} preloading".format(study))
                tic = time.time()
                train_dset_by_study[study].preload(count=args.num_workers)
                test_dset_by_study[study].preload(count=args.num_workers)
                print("Preloading {} took {}s".format(study, time.time() - tic))

    test_loaders_by_study = {
        study: torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        for study, test_dset in test_dset_by_study.items()
    }
    train_loaders_by_study = {
        study: torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        for study, train_dset in train_dset_by_study.items()
    }

    model = classifiers.versions[args.classifier_type](args, loadable_state_dict=None)
    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        print("Using dataparallel")
        model = nn.DataParallel(model)
    # New Optimizer
    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    # New scheduler
    if classifiers.scheduled[args.classifier_type]:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[5, 25],  # Start at 0.001 -> 0.0001, -> 0.00001
            gamma=0.1,
        )
    else:
        scheduler = None

    for batch_idx, batch in enumerate(utils.multi_key_infinite_iter(train_loaders_by_study)):
        if (batch_idx) % args.epoch_size == 0:
            if batch_idx > 0:
                if scheduler is not None:
                    scheduler.step()

                # Dump metrics, print time etc.
                for study in train_metrics.keys():
                    train_metrics[study]["loss"] /= args.epoch_size  # approximation.
                    train_metrics[study]["accuracy"] = sk_metrics.accuracy_score(ctrues[study], cpreds[study])
                    train_metrics[study]['precision'], train_metrics[study]['recall'], train_metrics[study]['f1'], train_metrics[study]['support'] = sk_metrics.precision_recall_fscore_support(ctrues[study], cpreds[study], labels=list(range(nclasses[study])))

                train_metrics_per_epoch.append(train_metrics)
                test_metrics = evaluate(args, model, test_loaders_by_study)
                # import pdb; pdb.set_trace()
                test_metrics_per_epoch.append(test_metrics)

                chk = utils.make_checkpoint(model, optimizer, batch_idx)
                torch.save(chk, os.path.join(output_dir, "last.checkpoint"))

                print(" ".join(
                    ["[{}/{}] t={:.2f}".format(batch_idx, args.max_batches, time.time() - tic)]
                    + [
                        "[{} train-loss={:.4f} train-acc={:.4f} test-acc={:.4f}]".format(
                            study,
                            train_metrics_per_epoch[-1][study]["loss"],
                            train_metrics_per_epoch[-1][study]["accuracy"],
                            test_metrics[study]["accuracy"],
                        ) for study in args.studies
                    ]
                ))

            # Create new metrics
            train_metrics = {
                study: {
                    'loss': 0.0,
                } for study in args.studies
            }
            nclasses = {}
            cpreds = {
                study: [] for study in args.studies
            }
            ctrues = {
                study: [] for study in args.studies
            }
            tic = time.time()
        if batch_idx == args.max_batches:
            break
        # Do the training use batch.
        for study, (x, _, _, cvec) in batch.items():
            nclasses[study] = len(args.meta['si2ci'][args.meta['s2i'][study]])
            N = x.shape[0]
            offset = min(args.meta['si2ci'][args.meta['s2i'][study]])
            cvec -= offset

            if args.cuda:
                x = x.cuda()
                cvec = cvec.cuda()
            study_vec = torch.tensor([args.meta['s2i'][study] for _ in range(N)], device=x.device).int()
            cpred = model(study_vec, x)
            loss = F.cross_entropy(cpred, cvec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_metrics[study]['loss'] += N * loss.item()
            ctrues[study].extend(cvec.cpu().tolist())
            cpreds[study].extend(torch.argmax(cpred.detach(), dim=1).cpu().tolist())

    torch.save(train_metrics_per_epoch, os.path.join(output_dir, "train_metrics_per_epoch.checkpoint"))
    torch.save(test_metrics_per_epoch, os.path.join(output_dir, "test_metrics_per_epoch.checkpoint"))

    if args.not_lazy:
        for study in train_dset_by_study.keys():
            if study in ["archi", "la5c"]:
                print("Starting unload")
                tic = time.time()
                train_dset_by_study[study].unload()
                test_dset_by_study[study].unload()
                print("Unloaded in {}s".format(time.time() - tic))

    return train_metrics_per_epoch, test_metrics_per_epoch


if __name__ == '__main__':
    args = get_args()
    print("Arguments parsed")
    args = classifiers.parse_model_specs(args)
    print("Model specs parsed")
    splits_by_study, meta = dataset.get_multi_study_splits(
        args.studies,
        args.outer_folds,
        1,
        args.dset_seed,
        random_outer=args.outer_frac,  # test, always. Not CV
        random_inner=0.0,  # No validation.
        masked=classifiers.masked[args.classifier_type],
        downsampled=args.downsampled,
        normalization=args.normalization,
        not_lazy=args.not_lazy
    )
    # import pdb; pdb.set_trace()
    print("Obtained splits")
    args.meta = meta
    args.wtree = constants.get_wtree()
    # Dump args to args.base_output/information.json
    torch.save(args, os.path.join(args.base_output, "args.checkpoint"))
    print("Dumped args")

    train_metrics_per_epoch_per_split = {}
    test_metrics_per_epoch_per_split = {}
    n_outer_splits = len(splits_by_study[args.studies[0]])
    tic = time.time()
    for split_idx in range(n_outer_splits):
        start = time.time()
        split_by_study = {}
        for study in args.studies:
            split_by_study[study] = splits_by_study[study][split_idx]
        split_dir = os.path.join(args.base_output, "outer_split{}".format(split_idx))
        os.makedirs(split_dir, exist_ok=True)
        (
            train_metrics_per_epoch_per_split[split_idx],
            test_metrics_per_epoch_per_split[split_idx],
        ) = run_single_split(
            args,
            split_by_study,
            epoch_size=args.epoch_size,
            output_dir=split_dir
        )

        metric_str = ""
        for study, metrics_dict in test_metrics_per_epoch_per_split[split_idx][-1].items():
            metric_str += "[{} last-acc={:.4f}] ".format(
                study,
                metrics_dict["accuracy"],
            )
        print("[Outer {}/{}] took {:.2f} {}".format(
            split_idx,
            n_outer_splits,
            time.time() - start,
            metric_str
        ))

    # Average out the metrics and dump them.
    average_last_test_metrics_by_study = {}
    for study in args.studies:
        average_last_test_metrics_by_study[study] = {
            metric_name: sum(
                test_metrics_per_epoch_per_split[split_idx][-1][study][metric_name] for split_idx in range(n_outer_splits)
            ) / n_outer_splits
            for metric_name in test_metrics_per_epoch_per_split[0][-1][study].keys()
        }
    torch.save(average_last_test_metrics_by_study, os.path.join(args.base_output, "last_metrics.checkpoint"))
    for study in args.studies:
        print("[{}] last-acc={:.4f}".format(
            study,
            average_last_test_metrics_by_study[study]["accuracy"],
        ))
    print("Total run time: {}".format(time.time() - tic))
