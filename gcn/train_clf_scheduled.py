"""
Perform train/test
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
from tensorboardX import SummaryWriter
from gcn.modules import (
    classifiers
)

import utils.utils as utils
from data import (
    dataset,
    constants
)


def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("name", type=str, choices=constants.nv_ids.keys(), help="dataset to use")  # noqa
    # misc
    parser.add_argument("-ns", "--subject_split", dest="subject_split", action="store_true", default=False, help="If set, train/test split is not by subject.")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    parser.add_argument("--downsampled", default=False, action="store_true", help="Use downsampled (to BASC template) data")
    parser.add_argument("--not_lazy", default=False, action="store_true", help="If provided, all data is loaded right away instead of being loaded on the fly")
    parser.add_argument("-nrm", "--normalization", dest="normalization", choices=['none', 'both', '0c', '11'], default="none", help="What kind of normalization to use")

    # compute
    parser.add_argument("--mode", choices=['train', 'test'], default='train')
    parser.add_argument("-chk", "--checkpoint", dest="checkpoint", default="", help="File where checkpoint is saved")

    # outputs
    parser.add_argument("--base_output", dest="base_output", default="gcn/outputs/clf/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="models", help="Subdirectory to save models")  # noqa

    # training
    # parser.add_argument("-nb", "--nbatches", dest="nbatches", type=int, metavar='<int>', default=200000, help="Number of batches to train on")  # noqa
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=200, help="Number of epochs to run this for")
    parser.add_argument("-brk", "--brk", dest="to_break", type=int, metavar='<int>', default=-1, help="Number of batches after which to breakpoint")  # noqa
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.01, help='Learning rate')  # noqa
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-4, help='Weight decay')  # noqa

    parser.add_argument("-clfc", "--classification_weight_contrast", dest="clfc", default=1.0, type=float, help="Weight to use on classification loss of contrast. Tweak to avoid class dependency")
    parser.add_argument("-clft", "--classification_weight_task", dest="clft", default=0, type=float, help="Weight to use on classification loss of task. Tweak to avoid class dependency")
    parser.add_argument("-clfs", "--classification_weight_study", dest="clfs", default=0, type=float, help="Weight to use on classification loss of study. Tweak to avoid class dependency")

    parser.add_argument('-ve', '--validate_every', type=int, default=5, help='How often to validate')  # noqa
    parser.add_argument('-se', '--save_every', type=int, default=10, help='How often to save during training')  # noqa
    # parser.add_argument('-pe', '--print_every', type=int, default=20, help='How often to print losses during training')  # noqa

    parser.add_argument('-ct', "--classifier_type", type=str, default='fgl0', choices=classifiers.versions.keys(), help="What classifier version to use")
    parser.add_argument('-nr', '--nregions', type=int, default=8, help='How many regions to consider if classifier type is r0')  # noqa
    args = parser.parse_args()

    args.names = args.name
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Get run_code
    # Check if base_output exists
    args.base_output = os.path.join(args.base_output, args.classifier_type)
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    print("Using run_code: {}".format(args.run_code))
    # If directory doesn't exist, create it
    args.tensorboard_dir = os.path.join(args.base_output, args.run_code, args.tensorboard_dir)  # noqa
    args.model_dir = os.path.join(args.base_output, args.run_code, args.model_dir)  # noqa

    directories_needed = [args.tensorboard_dir, args.model_dir]

    for dir_name in directories_needed:
        os.makedirs(dir_name, exist_ok=True)
    return args


def test(args, datasets, loaders, model, writer, batch_idx=-1, prefix="test", save_on_batch=None):
    start = time.time()
    model = model.eval()
    save_on_batch = random.randint(0, 10) if save_on_batch is None else save_on_batch
    with torch.no_grad():
        for s, loader in loaders.items():
            not_dumped_images = True
            losses = defaultdict(lambda: [])
            for bidx, (x, _, _, cvec) in enumerate(loader):
                batch_size = x.shape[0]

                if args.cuda:
                    x = x.cuda()
                    # svec = svec.cuda()
                    # tvec = tvec.cuda()
                    cvec = cvec.cuda()

                cpred = model(x)
                losses["contrast acc"].append((torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item())
                losses["contrast ce"].append(F.cross_entropy(cpred, cvec).item())
        for k, v in losses.items():
            print("[{}-{} {}][{:.2f}]{} = {}".format(prefix, s, batch_idx, time.time() - start, k, np.mean(v)))

        writer.add_scalars(
            prefix,
            {s + "-" + k: np.mean(v) for k, v in losses.items()},
            batch_idx
        )

    model = model.train()


def train_single_dataset(args, train_datasets, train_loaders, test_datasets, test_loaders, model, writer, prefix="train", checkpoint=None):
    assert(len(train_datasets) == 1)
    study = list(train_datasets.keys())[0]
    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    if classifiers.scheduled[args.classifier_type]:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 250, 500],  # Start at 0.01, -> 0.001, -> 0.0001 -> 0.00001
            gamma=0.1,
        )
    else:
        scheduler = None
    if checkpoint:
        optimizier.load_state_dict(checkpoint['optimizier'])
        start_batch = load_state_dict(checkpoint['start_batch'])
    else:
        start_batch = 0
    to_break = args.to_break
    batch_count = -1
    for eidx in range(args.epochs):
        if scheduler is not None:
            scheduler.step()  # Increment count by 1
        start = time.time()
        forward_prop_time = 0.0
        backward_prop_time = 0.0
        gradient_penalty_time = 0.0
        cuda_transfer_time = 0.0
        epoch_losses = defaultdict(lambda: [])
        for bidx, (x, _, _, cvec) in enumerate(train_loaders[study]):
            bstart = time.time()
            batch_count += 1
            if batch_count == to_break:
                pdb.set_trace()

            temp_time = time.time()
            N = x.shape[0]
            if args.cuda:
                x = x.cuda()
                cvec = cvec.cuda()
            cuda_transfer_time += time.time() - temp_time

            temp_time = time.time()
            cpred = model(x)
            loss = F.cross_entropy(cpred, cvec)
            acc = (torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item()
            epoch_losses["{} ce".format(study)].append(loss.item())
            epoch_losses["{} acc".format(study)].append(acc)
            forward_prop_time += time.time() - temp_time

            temp_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            # for pn, p in model.named_parameters():
            #     if p.requires_grad and not("bias" in pn):
            #         epoch_losses["norm-{}".format(pn)] = p.detach().norm().item()
            #         epoch_losses["norm-grad-{}".format(pn)] = p.grad.detach().norm().item()
                    # if torch.isnan(p.grad).any():
                    #     pdb.set_trace()
            optimizer.step()
            backward_prop_time += time.time() - temp_time
            if bidx == 0:
                print("Batch {}/{} took {}s".format(bidx, len(train_loaders[study]), time.time() - bstart))
            del x, cvec, loss
        writer.add_scalars(
            prefix,
            {k: np.mean(v) for k, v in epoch_losses.items()},
            eidx
        )
        print("[{}: {} / {}]: loss={:.3f} acc={:.3f} cuda={:.2f} forw={:.2f} back={:.2f} total={:.2f}".format(prefix, eidx, args.epochs, np.mean(epoch_losses["{} ce".format(study)]), np.mean(epoch_losses["{} acc".format(study)]), cuda_transfer_time, forward_prop_time, backward_prop_time, time.time() - start))

        ######
        # Val
        ######
        if (utils.periodic_integer_delta(eidx, every=args.validate_every, start=-1) or (eidx == (args.epochs - 1))):
            test(args, test_datasets, test_loaders, model, writer, batch_idx=eidx, prefix="val")
            model = model.train()

        ######
        # SAVE
        ######
        if (utils.periodic_integer_delta(eidx, every=args.save_every, start=-1) or (eidx == (args.epochs - 1))):
            checkpoint = {}
            checkpoint['base_dir'] = os.path.join(args.base_output, args.run_code)  # Makes retrieval of hyperparameters easy
            checkpoint['eidx'] = eidx
            checkpoint['optimizer'] = optimizer.state_dict()
            if args.dataparallel:
                model = model.module  # Potentially slow.
                checkpoint['model'] = model.state_dict()
                model = nn.DataParallel(model)
            else:
                checkpoint['model'] = model.state_dict()
            checkpoint_path = os.path.join(args.model_dir, "model_epoch{}{}".format(eidx, ".checkpoint"))
            torch.save(checkpoint, checkpoint_path)
            del checkpoint_path
            del checkpoint

    pass


if __name__ == '__main__':
    # get args
    args = get_args()
    # get datasets
    cpu_count = multiprocessing.cpu_count() // 3
    trn, tst, meta, train_loaders, test_loaders = dataset.get_dataloaders(
        studies=args.names,
        subject_split=args.subject_split,
        debug=args.debug,
        batch_size=args.batch_size,
        num_workers=cpu_count,
        masked=classifiers.masked[args.classifier_type],
        downsampled=args.downsampled,
        normalization=args.normalization,
        not_lazy=args.not_lazy,
    )
    print("Datasets instantiated (lazy loading due to memory concerns)")
    args.meta = meta
    args.wtree = constants.get_wtree()
    # import pdb; pdb.set_trace()
    print("Ward tree loaded")
    if args.checkpoint != "":
        checkpoint = torch.load(args.checkpoint)
        model_state_dict = checkpoint['model']
    else:
        checkpoint = None
        model_state_dict = None
    # make model
    model = classifiers.versions[args.classifier_type](args, loadable_state_dict=model_state_dict)

    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        print("Using dataparallel")
        model = nn.DataParallel(model)
    print("Model instantiated")
    # dump argparse
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    utils.dump_everything(args)
    # call train/test
    if args.mode == 'train':
        train_single_dataset(args, trn, train_loaders, tst, test_loaders, model, writer, prefix="train", checkpoint=checkpoint)
    elif args.mode == 'test':
        test(args, tst, test_loaders, model, writer)
    print("Used run_code: {}".format(args.run_code))
