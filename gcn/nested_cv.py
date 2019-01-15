"""
Perform train/test on nested folds
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
from tensorboardX import SummaryWriter
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
    parser.add_argument("-ok", "--outer_folds", type=int, metavar='<int>', default=5, help="Number of outer folds")  # noqa
    parser.add_argument("-ik", "--inner_folds", type=int, metavar='<int>', default=5, help="Number of inner folds (runs per outer fold)")  # noqa
    parser.add_argument("-rf", "--inner_frac", type=float, metavar='<int>', default=0.2, help="Fraction of inner fold used for validation set")  # noqa
    parser.add_argument("-ds", "--dset_seed", type=int, metavar='<int>', default=1337, help="Seed used for dataset")  # noqa

    # Data parameters... don't mess with this.
    parser.add_argument("--downsampled", default=False, action="store_true", help="Use downsampled (to BASC template) data")
    parser.add_argument("--not_lazy", default=False, action="store_true", help="If provided, all data is loaded right away instead of being loaded on the fly")
    parser.add_argument("-nrm", "--normalization", dest="normalization", choices=['none', 'both', '0c', '11'], default="none", help="What kind of normalization to use")
    parser.add_argument("--dbg", dest="debug", default=False, action="store_true", help="if true, then set to 40 datapoints in each dataset.")  # noqa

    parser.add_argument("--base_output", dest="base_output", default="gcn/outputs/kfold_clf/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa

    # System parameters
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    # Choose the learning parameters
    parser.add_argument("-maxe", "--max_epochs", dest="max_epochs", type=int, default=30, help="Maximum number of epochs to run for")
    parser.add_argument("-mine", "--min_epochs", dest="min_epochs", type=int, default=10, help="Minimum number of epochs to run for")
    parser.add_argument("-pat", "--patience", dest="patience", type=int, default=3, help="# of epochs of patience")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa

    # Choose the model
    parser.add_argument('-ct', "--classifier_type", type=str, default='fc', choices=classifiers.versions.keys(), help="What classifier version to use")
    parser.add_argument('-nr', '--nregions', type=int, default=8, help='How many regions to consider if classifier type is r0')  # noqa
    parser.add_argument("--no_opt", dest="no_opt", default=False, action="store_true", help="Disable packing/padding optimization")  # noqa

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
            if args.cuda and not(args.dataparallel):
                # If dataparallel, then nn.DataParallel will automatically send stuff to the correct device
                x = x.cuda()
                cvec = cvec.cuda()
            elif args.cuda:
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


def train_one_epoch(args, model, optimizer, loader):
    nclasses = len(args.meta['c2i'])
    metrics = {
        'loss': 0.0,
    }
    tic = time.time()
    ctrues = []
    cpreds = []
    for bidx, (x, _, _, cvec) in enumerate(loader):
        N = x.shape[0]
        if args.cuda and not(args.dataparallel):
            # If dataparallel, then nn.DataParallel will automatically send stuff to the correct device
            x = x.cuda()
            cvec = cvec.cuda()
        elif args.cuda:
            cvec = cvec.cuda()
        cpred = model(x)
        loss = F.cross_entropy(cpred, cvec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics['loss'] += N * loss.item()
        ctrues.extend(cvec.cpu().tolist())
        cpreds.extend(torch.argmax(cpred.detach(), dim=1).cpu().tolist())
    metrics['loss'] /= len(loader)
    metrics['cm'] = sk_metrics.confusion_matrix(ctrues, cpreds, labels=list(range(nclasses)))
    metrics['accuracy'] = sk_metrics.accuracy_score(ctrues, cpreds)
    metrics['precision'], metrics['recall'], metrics['f1'], metrics['support'] = sk_metrics.precision_recall_fscore_support(ctrues, cpreds, labels=list(range(nclasses)))
    metrics['time'] = time.time() - tic
    return metrics


def validation_improvement(new_val_metrics, old_val_metrics):
    return (new_val_metrics['accuracy'] > old_val_metrics['accuracy']) or (new_val_metrics['loss'] < old_val_metrics['loss'])


def make_checkpoint(model, optimizer, epoch):
    chk = {}
    if isinstance(model, nn.DataParallel):
        chk['model'] = model.module.state_dict()
    else:
        chk['model'] = model.state_dict()
    chk['optimizer'] = optimizer.state_dict()
    chk['epoch'] = epoch
    return chk


def single_split_run_with_patience_stopping(args, split, output_dir=".", patience=3):
    test_dset = split['test']
    if args.debug:
        test_dset.n = DEBUG_N
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    all_train_metrics_per_cv = {}
    all_val_metrics_per_cv = {}
    all_test_metrics_per_cv = {}
    best_train_metrics_per_cv = {}
    best_val_metrics_per_cv = {}
    best_test_metrics_per_cv = {}
    for cv_idx, (dsets) in enumerate(split['splits']):
        split_tic = time.time()
        train_dset = dsets['train']
        val_dset = dsets['val']
        # Make dataloaders
        if args.debug:
            train_dset.n = DEBUG_N
            val_dset.n = DEBUG_N
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        # New Model
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
                milestones=[5, 25],  # Start at 0.01 -> 0.001, -> 0.0001
                gamma=0.1,
            )
        else:
            scheduler = None

        cv_output_dir = os.path.join(output_dir, "inner_split{}".format(cv_idx))
        os.makedirs(cv_output_dir, exist_ok=True)

        cur_patience = patience

        all_train_metrics_per_cv[cv_idx] = []
        all_val_metrics_per_cv[cv_idx] = []
        all_test_metrics_per_cv[cv_idx] = []

        best_train_metrics_per_cv[cv_idx] = None
        best_val_metrics_per_cv[cv_idx] = None
        best_test_metrics_per_cv[cv_idx] = None

        old_val_metrics = None
        for epoch in range(args.max_epochs):
            epoch_tic = time.time()
            if scheduler is not None:
                scheduler.step()
            train_metrics = train_one_epoch(args, model, optimizer, train_loader)
            val_metrics = evaluate(args, model, val_loader)
            test_metrics = evaluate(args, model, test_loader)
            improvement = (epoch < args.min_epochs) or (old_val_metrics is None) or validation_improvement(val_metrics, old_val_metrics)
            if improvement:  # If improvement - save model, reset patience
                # Save checkpoint.
                is_best = (best_val_metrics_per_cv[cv_idx] is None) or val_metrics['accuracy'] >= best_val_metrics_per_cv[cv_idx]['accuracy']
                if is_best:
                    chk = make_checkpoint(model, optimizer, epoch)
                    torch.save(chk, os.path.join(cv_output_dir, "best.checkpoint"))
                    best_train_metrics_per_cv[cv_idx] = train_metrics
                    best_val_metrics_per_cv[cv_idx] = val_metrics
                    best_test_metrics_per_cv[cv_idx] = test_metrics
                cur_patience = patience
            else:  # If no improvement, drop patience
                cur_patience -= 1
            if cur_patience == 0:
                break  # We're done.
            # Update old_metrics
            old_val_metrics = val_metrics
            all_train_metrics_per_cv[cv_idx].append(train_metrics)
            all_val_metrics_per_cv[cv_idx].append(val_metrics)
            all_test_metrics_per_cv[cv_idx].append(test_metrics)
            print(
                "[Epoch {}/{}] train-loss={:.4f} val-acc={:.4f} test-acc={:.4f} time={:.2f}".format(
                    epoch,
                    args.max_epochs,
                    train_metrics["loss"],
                    val_metrics["accuracy"],
                    test_metrics["accuracy"],
                    time.time() - epoch_tic,
                )
            )
        # Save checkpoint
        torch.save(chk, os.path.join(cv_output_dir, "last.checkpoint"))
        metrics = {
            'train': all_train_metrics_per_cv[cv_idx],  # list of dicts
            'val': all_val_metrics_per_cv[cv_idx],  # list of dicts
            'test': all_test_metrics_per_cv[cv_idx],  # list of dicts.
        }
        torch.save(metrics, os.path.join(cv_output_dir, "metrics.checkpoint"))
        print("[Inner {}/{}] took {:.2f}s, acc={:.4f}".format(
            cv_idx, len(split['splits']), time.time() - split_tic, best_test_metrics_per_cv[cv_idx]["accuracy"]
        ))
    average_test_metrics = {
        metric_name: sum(
            [
                best_test_metrics_per_cv[cv_idx][metric_name]
                for cv_idx in all_test_metrics_per_cv.keys()
            ]
        ) / len(split['splits'])
        for metric_name in best_test_metrics_per_cv[0].keys()
    }
    torch.save(average_test_metrics, os.path.join(output_dir, "average_best_metrics.checkpoint"))
    return all_train_metrics_per_cv, all_val_metrics_per_cv, all_test_metrics_per_cv, average_test_metrics


if __name__ == '__main__':
    args = get_args()
    splits, meta = dataset.get_splits(
        args.study,
        args.outer_folds,
        args.inner_folds,
        args.dset_seed,
        random_inner=args.inner_frac,
        masked=classifiers.masked[args.classifier_type],
        downsampled=args.downsampled,
        normalization=args.normalization,
        not_lazy=args.not_lazy
    )
    args.meta = meta
    args.wtree = constants.get_wtree()
    # Dump args to args.base_output/information.json
    torch.save(args, os.path.join(args.base_output, "args.checkpoint"))
    all_train_metrics_per_cv_per_split = {}
    all_val_metrics_per_cv_per_split = {}
    all_test_metrics_per_cv_per_split = {}
    all_average_best_metrics_per_split = {}
    tic = time.time()
    for split_idx, split in enumerate(splits):
        start = time.time()
        split_dir = os.path.join(args.base_output, "outer_split{}".format(split_idx))
        (
            all_train_metrics_per_cv_per_split[split_idx],
            all_val_metrics_per_cv_per_split[split_idx],
            all_test_metrics_per_cv_per_split[split_idx],
            all_average_best_metrics_per_split[split_idx],
        ) = single_split_run_with_patience_stopping(
            args,
            split,
            output_dir=split_dir,
            patience=args.patience,
        )
        print("[Outer {}/{}] took {:.2f}, acc={:.4f}".format(
            split_idx,
            len(splits),
            time.time() - start,
            all_average_best_metrics_per_split[split_idx]["accuracy"]
        ))
    # Average out test metrics and dump.
    average_test_metrics = {
        metric_name: sum(
            [
                all_average_best_metrics_per_split[split_idx][metric_name]
                for split_idx in all_average_best_metrics_per_split.keys()
            ]
        ) / len(splits)
        for metric_name in all_average_best_metrics_per_split[0].keys()
    }
    torch.save(average_test_metrics, os.path.join(args.base_output, "metrics.checkpoint"))
    print('Average accuracy={:.4f}'.format(average_test_metrics["accuracy"]))
    print("Total run time: {}".format(time.time() - tic))
