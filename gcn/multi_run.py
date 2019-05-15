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
from gcn.modules import (
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
    parser.add_argument("study", type=str, choices=constants.nv_ids.keys(), help="dataset to use")  # noqa
    parser.add_argument("-ok", "--outer_folds", type=int, metavar='<int>', default=10, help="Number of outer folds")  # noqa
    parser.add_argument("-df", "--outer_frac", type=float, metavar='<int>', default=0.3, help="Fraction of data to use as test in each outer fold")  # noqa
    parser.add_argument('-tf', '--training_frac', type=float, default=0.7, help='fraction of data to use for training.')
    parser.add_argument("-ds", "--dset_seed", type=int, metavar='<int>', default=1337, help="Seed used for dataset")  # noqa

    # Data parameters... don't mess with this.
    parser.add_argument("--downsampled", default=False, action="store_true", help="Use downsampled (to BASC template) data")
    parser.add_argument("--not_lazy", default=False, action="store_true", help="If provided, all data is loaded right away instead of being loaded on the fly")
    parser.add_argument("-nrm", "--normalization", dest="normalization", choices=['none', 'both', '0c', '11'], default="none", help="What kind of normalization to use")
    parser.add_argument("--dbg", dest="debug", default=False, action="store_true", help="if true, then set to 40 datapoints in each dataset.")  # noqa

    parser.add_argument("--base_output", dest="base_output", default="gcn/outputs/multi_run/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa

    # System parameters
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    # Choose the learning parameters
    parser.add_argument("-maxe", "--max_epochs", dest="max_epochs", type=int, default=30, help="Maximum number of epochs to run for")
    parser.add_argument("-mine", "--min_epochs", dest="min_epochs", type=int, default=10, help="Minimum number of epochs to run for")
    # parser.add_argument("-thr", "--loss_thr", dest="loss_thr", type=float, default=1e-4, help="If training loss change is less than this, break")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Choose the model
    parser.add_argument('-ct', "--classifier_type", type=str, default='fc', help="What classifier version to use",)  # choices=classifiers.versions.keys(),)
    parser.add_argument('-nr', '--nregions', type=int, default=8, help='How many regions to consider if classifier type is r0')  # noqa

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


def evaluate(args, model, loader):
    nclasses = len(args.meta['c2i'])
    metrics = {
        'loss': 0.0,
    }
    model = model.eval()
    tic = time.time()
    ctrues = []
    cpreds = []
    with torch.no_grad():
        for bidx, (x, _, _, cvec) in enumerate(loader):
            N = x.shape[0]
            if args.cuda:
                # If dataparallel, then nn.DataParallel will automatically send stuff to the correct device
                x = x.cuda()
                cvec = cvec.cuda()
            cpred = model(x)
            metrics['loss'] += N * F.cross_entropy(cpred, cvec).item()
            ctrues.extend(cvec.cpu().tolist())
            cpreds.extend(torch.argmax(cpred.detach(), dim=1).cpu().tolist())
    model = model.train()
    metrics['loss'] /= len(loader)
    metrics['cm'] = sk_metrics.confusion_matrix(ctrues, cpreds, labels=list(range(nclasses)))
    metrics['accuracy'] = sk_metrics.accuracy_score(ctrues, cpreds)
    metrics['precision'], metrics['recall'], metrics['f1'], metrics['support'] = sk_metrics.precision_recall_fscore_support(ctrues, cpreds, labels=list(range(nclasses)))
    metrics['time'] = time.time() - tic
    return metrics


to_break = False
def train_one_epoch(args, model, optimizer, loader, tobreak=False):
    nclasses = len(args.meta['c2i'])
    metrics = {
        'loss': 0.0,
    }
    tic = time.time()
    ctrues = []
    cpreds = []
    for bidx, (x, _, _, cvec) in enumerate(loader):
        N = x.shape[0]
        if args.cuda:
            # If dataparallel, then nn.DataParallel will automatically send stuff to the correct device
            x = x.cuda()
            cvec = cvec.cuda()
        cpred = model(x)
        loss = F.cross_entropy(cpred, cvec)
        optimizer.zero_grad()
        loss.backward()
        if to_break:
            import pdb; pdb.set_trace()
        optimizer.step()
        metrics['loss'] += N * loss.item()
        ctrues.extend(cvec.cpu().tolist())
        cpreds.extend(torch.argmax(cpred.detach(), dim=1).cpu().tolist())
    metrics['loss'] /= len(loader)
    metrics['cm'] = sk_metrics.confusion_matrix(ctrues, cpreds, labels=list(range(nclasses))).astype(np.float32)
    metrics['accuracy'] = sk_metrics.accuracy_score(ctrues, cpreds)
    metrics['precision'], metrics['recall'], metrics['f1'], metrics['support'] = sk_metrics.precision_recall_fscore_support(ctrues, cpreds, labels=list(range(nclasses)))
    metrics['time'] = time.time() - tic
    return metrics


def run_single_split(args, split, output_dir="."):
    train_metrics_per_epoch = []
    test_metrics_per_epoch = []
    best_test_metrics = None
    test_dset = split['test']
    train_dset = split['splits'][0]['train']
    assert(len(split['splits']) == 1)  # Just one split.
    if args.debug:
        test_dset.n = DEBUG_N
        train_dset.n = DEBUG_N
    print("Starting preloading")
    tic = time.time()
    if args.not_lazy:
        train_dset.preload(count=args.num_workers)
        test_dset.preload(count=args.num_workers)
    print("Preloading took {}s".format(time.time() - tic))

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    model = classifiers.versions[args.classifier_type](args, loadable_state_dict=None)
    # import pdb; pdb.set_trace()
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

    old_loss = None
    min_loss = None
    for epoch in range(args.max_epochs):
        epoch_tic = time.time()
        if scheduler is not None:
            scheduler.step()
        train_metrics = train_one_epoch(args, model, optimizer, train_loader)  # , tobreak=(epoch==(0)))
        train_metrics_per_epoch.append(train_metrics)

        test_metrics = evaluate(args, model, test_loader)
        test_metrics_per_epoch.append(test_metrics)

        is_best = (best_test_metrics is None) or test_metrics['accuracy'] >= best_test_metrics['accuracy']
        if is_best:
            chk = utils.make_checkpoint(model, optimizer, epoch)
            torch.save(chk, os.path.join(output_dir, "best.checkpoint"))
            best_test_metrics = test_metrics
        chk = utils.make_checkpoint(model, optimizer, epoch)
        torch.save(chk, os.path.join(output_dir, "last.checkpoint"))
        # breaking logic: Not yet implemented
        # new_loss = train_metrics['loss']
        # if min_loss is None:
        #     min_loss = new_loss
        # else:
        #     min_loss = min(new_loss, min_loss)
        #
        # to_break = (
        #     (epoch > args.min_epochs) and
        #     (old_loss is not None) and
        #     ((abs(new_loss - old_loss) < args.loss_thr)) and
        #     ((abs(new_loss - min_loss) < args.loss_thr))
        # )
        # old_loss = new_loss
        # if to_break:
        #     break
        print(
            "[Epoch {}/{}] train-loss={:.4f} train-acc={:.4f} test-acc={:.4f} time={:.2f}".format(
                epoch,
                args.max_epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                test_metrics["accuracy"],
                time.time() - epoch_tic,
            )
        )

    torch.save(train_metrics_per_epoch, os.path.join(output_dir, "train_metrics_per_epoch.checkpoint"))
    torch.save(test_metrics_per_epoch, os.path.join(output_dir, "test_metrics_per_epoch.checkpoint"))
    torch.save(best_test_metrics, os.path.join(output_dir, "best_test_metrics.checkpoint"))
    print("Starting unload")
    tic = time.time()
    if args.not_lazy:
        train_dset.unload()
        test_dset.unload()
    print("Unloaded in {}s".format(time.time() - tic))
    return train_metrics_per_epoch, test_metrics_per_epoch, best_test_metrics


if __name__ == '__main__':
    args = get_args()
    print("Arguments parsed")
    args = classifiers.parse_model_specs(args)
    print("Model specs parsed")
    splits, meta = dataset.get_splits(
        args.study,
        args.outer_folds,
        1,
        args.dset_seed,
        random_outer=args.outer_frac,  # test, always. Not CV
        random_inner=1 - args.outer_frac - args.training_frac,  # No validation.
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
    best_metrics_per_split = {}
    tic = time.time()
    for split_idx, split in enumerate(splits):
        start = time.time()
        split_dir = os.path.join(args.base_output, "outer_split{}".format(split_idx))
        os.makedirs(split_dir, exist_ok=True)
        (
            train_metrics_per_epoch_per_split[split_idx],
            test_metrics_per_epoch_per_split[split_idx],
            best_metrics_per_split[split_idx],
        ) = run_single_split(args, split, split_dir)
        print("[Outer {}/{}] took {:.2f}, best-acc={:.4f} last-acc={:.4f}".format(
            split_idx,
            len(splits),
            time.time() - start,
            best_metrics_per_split[split_idx]["accuracy"],
            test_metrics_per_epoch_per_split[split_idx][-1]["accuracy"],
        ))

    # Average out test metrics and dump.
    average_last_test_metrics = {
        metric_name: sum(
            [
                test_metrics_per_epoch_per_split[split_idx][-1][metric_name]
                for split_idx in test_metrics_per_epoch_per_split.keys()
            ]
        ) / len(splits)
        for metric_name in test_metrics_per_epoch_per_split[0][-1].keys()
    }
    torch.save(average_last_test_metrics, os.path.join(args.base_output, "last_metrics.checkpoint"))
    print('Average last accuracy={:.4f}'.format(average_last_test_metrics["accuracy"]))

    # Average out test metrics and dump.
    average_best_test_metrics = {
        metric_name: sum(
            [
                best_metrics_per_split[split_idx][metric_name]
                for split_idx in best_metrics_per_split.keys()
            ]
        ) / len(splits)
        for metric_name in best_metrics_per_split[0].keys()
    }
    torch.save(average_best_test_metrics, os.path.join(args.base_output, "best_metrics.checkpoint"))
    print('Average best accuracy={:.4f}'.format(average_best_test_metrics["accuracy"]))

    print("Total run time: {}".format(time.time() - tic))
