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
from tensorboardX import SummaryWriter
from conv.modules import (
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
    parser.add_argument("names", type=str, nargs='+', choices=constants.nv_ids.keys(), help="datasets to use")  # noqa
    # misc
    parser.add_argument("-ns", "--subject_split", dest="subject_split", action="store_true", default=False, help="If set, train/test split is not by subject.")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")

    # compute
    parser.add_argument("--mode", choices=['train', 'test'], default='train')
    parser.add_argument("-chk", "--checkpoint", dest="checkpoint", default="", help="File where checkpoint is saved")
    # outputs
    parser.add_argument("--base_output", dest="base_output", default="conv/outputs/shacgan/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
    parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="models", help="Subdirectory to save models")  # noqa

    # training
    parser.add_argument("-nb", "--nbatches", dest="nbatches", type=int, metavar='<int>', default=200000, help="Number of batches to train on")  # noqa
    parser.add_argument("-brk", "--brk", dest="to_break", type=int, metavar='<int>', default=-1, help="Number of batches after which to breakpoint")  # noqa
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0.0, help='Weight decay')  # noqa

    parser.add_argument("-clfc", "--classification_weight_contrast", dest="clfc", default=2.0, type=float, help="Weight to use on classification loss of contrast. Tweak to avoid class dependency")
    parser.add_argument("-clft", "--classification_weight_task", dest="clft", default=1.0, type=float, help="Weight to use on classification loss of task. Tweak to avoid class dependency")
    parser.add_argument("-clfs", "--classification_weight_study", dest="clfs", default=0.5, type=float, help="Weight to use on classification loss of study. Tweak to avoid class dependency")

    parser.add_argument('-ve', '--validate_every', type=int, default=100, help='How often to validate')  # noqa
    parser.add_argument('-se', '--save_every', type=int, default=1000, help='How often to save during training')  # noqa
    # parser.add_argument('-pe', '--print_every', type=int, default=20, help='How often to print losses during training')  # noqa

    parser.add_argument('-ct', "--classifier_type", type=str, default='0', choices=classifiers.versions.keys(), help="What classifier version to use")
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Get run_code
    # Check if base_output exists
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
            for bidx, (x, svec, tvec, cvec) in enumerate(loader):
                batch_size = x.shape[0]

                if args.cuda:
                    x = x.cuda()
                    svec = svec.cuda()
                    tvec = tvec.cuda()
                    cvec = cvec.cuda()

                spred, tpred, cpred = model.discriminator(xin, predict_s=r_labels, svec=svec, tvec=tvec)
                losses[tag + "study acc"].append((torch.argmax(spred.detach(), dim=1) == svec).float().mean().item())
                losses[tag + "study ce"].append(F.cross_entropy(spred, svec).item())
                losses[tag + "task acc"].append((torch.argmax(tpred.detach(), dim=1) == tvec).float().mean().item())
                losses[tag + "task ce"].append(F.cross_entropy(tpred, tvec).item())
                losses[tag + "contrast acc"].append((torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item())
                losses[tag + "contrast ce"].append(F.cross_entropy(cpred, cvec).item())

                # Log stuff
            writer.add_scalars(
                (prefix + "-" if prefix != "" else "") + s,
                {k: np.mean(v) for k, v in losses.items()},
                batch_idx
            )
    model = model.train()
    print("{}testing took {}s".format(prefix + " " if prefix != "" else "", time.time() - start))


def train(args, train_datasets, train_loaders, test_datasets, test_loaders, model, writer, prefix="train", checkpoint=None):
    params = list(model.parameters())
    optimizer = optim.Adam(
        gen_params,
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    if checkpoint:
        optimizier.load_state_dict(checkpoint['optimizier'])
        start_batch = load_state_dict(checkpoint['start_batch'])
    else:
        start_batch = 0
    to_break = args.to_break

    # Unused variable.
    for batch_idx, batch in enumerate(utils.multi_key_infinite_iter(train_loaders)):
        start = time.time()
        forward_prop_time = 0.0
        backward_prop_time = 0.0
        gradient_penalty_time = 0.0
        cuda_transfer_time = 0.0
        writing_time = 0.0
        if batch_idx == args.nbatches:  # Termination condition
            break
        batch_items = list(batch.items())
        random.shuffle(batch_items)
        batch_losses = {}
        for study, (x, svec, tvec, cvec) in batch_items:
            losses = {}
            if ((batch_idx == to_break)):
                pdb.set_trace()
            batch_size = x.shape[0]
            # f_labels = torch.zeros((batch_size), dtype=torch.float, device=args.device)

            temp_time = time.time()
            if args.cuda:
                x = x.cuda()
                svec = svec.cuda()
                tvec = tvec.cuda()
                cvec = cvec.cuda()

            cuda_transfer_time += time.time() - temp_time

            ####################
            # DISCRIMINATOR
            ####################
            # Discriminator updates
            spred, tpred, cpred = model(x)
            # pdb.set_trace()
            L_clf_s = F.cross_entropy(spred, svec)
            L_clf_t = F.cross_entropy(tpred, tvec)
            L_clf_c = F.cross_entropy(cpred, cvec)
            disc_real_loss = (L_clf_s + L_clf_t + L_clf_c)
            acc_s = (torch.argmax(spred.detach(), dim=1) == svec).float().mean().item()
            acc_t = (torch.argmax(tpred.detach(), dim=1) == tvec).float().mean().item()
            acc_c = (torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item()

            forward_prop_time += time.time() - temp_time
            temp_time = time.time()

            losses["real-" + "s acc"] = acc_s
            losses["real-" + "s ce"] = L_clf_s.item()
            losses["real-" + "t acc"] = acc_t
            losses["real-" + "t ce"] = L_clf_t.item()
            losses["real-" + "c acc"] = acc_c
            losses["real-" + "c ce"] = L_clf_c.item()
            disc_real_loss.backward()
            for pn, p in model.named_parameters():
                if p.requires_grad and not("bias" in pn):
                    losses["norm-disc-{}".format(pn)] = p.detach().norm().item()
                    losses["norm-disc-grad-{}".format(pn)] = p.grad.detach().norm().item()
                    # if torch.isnan(p.grad).any():
                    #     pdb.set_trace()
            disc_optimizer.step()
            backward_prop_time += time.time() - temp_time
            for k, v in losses.items():
                batch_losses[study + "-" + k] = v

        temp_time = time.time()
        writer.add_scalars(
            (prefix),
            {k: v for k, v in batch_losses.items()},
            batch_idx
        )
        writing_time += time.time() - temp_time
        # pdb.set_trace()
        print("[{} / {}]: writing={:.2f} cuda={:.2f} forw={:.2f} back={:.2f} total={:.2f}".format(batch_idx, args.nbatches, writing_time, cuda_transfer_time, forward_prop_time, backward_prop_time, time.time() - start))
        ######
        # Val
        ######
        if (utils.periodic_integer_delta(batch_idx, every=args.validate_every, start=-1) or (batch_idx == (args.nbatches - 1))):
            test(args, test_datasets, test_loaders, model, writer, batch_idx=batch_idx, prefix="val")
            model = model.train()

        ######
        # SAVE
        ######
        if (utils.periodic_integer_delta(batch_idx, every=args.save_every, start=-1) or (batch_idx == (args.nbatches - 1))):
            checkpoint = {}
            checkpoint['base_dir'] = os.path.join(args.base_output, args.run_code)  # Makes retrieval of hyperparameters easy
            # checkpoint['args'].meta = {k: v for k, v in args.meta.items()}
            checkpoint['batch_idx'] = batch_idx
            checkpoint['disc_optimizer'] = disc_optimizer.state_dict()
            # checkpoint['enc_optimizer'] = enc_optimizer.state_dict()
            checkpoint['gen_optimizer'] = gen_optimizer.state_dict()
            if args.dataparallel:
                model = model.module  # Potentially slow.
                checkpoint['model'] = model.state_dict()
                model = nn.DataParallel(model)
            else:
                checkpoint['model'] = model.state_dict()
            checkpoint_path = os.path.join(args.model_dir, "model_batch{}{}".format(batch_idx, ".checkpoint"))
            torch.save(checkpoint, checkpoint_path)
            del checkpoint_path
            del checkpoint

if __name__ == '__main__':
    # argparse
    args = get_args()

    # datasets
    cpu_count = 2 * multiprocessing.cpu_count() // 3
    trn, tst, meta, train_loaders, test_loaders = dataset.get_dataloaders(
        studies=args.names,
        subject_split=args.subject_split,
        debug=args.debug,
        batch_size=args.batch_size,
        num_workers=cpu_count
    )
    args.meta = meta

    # create model
    if args.checkpoint != "":
        checkpoint = torch.load(args.checkpoint)
        model_state_dict = checkpoint['model']
    else:
        checkpoint = None
        model_state_dict = None
    model = classifiers.versions[args.classifier_type](
        args,
        loadable_state_dict=model_state_dict
    )
    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

    # dump argparse
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    utils.dump_everything(args)

    # train/test
    if args.mode == 'train':
        train(args, trn, train_loaders, tst, test_loaders, model, writer, prefix="train", checkpoint=checkpoint)
    elif args.mode == 'test':
        test(args, tst, test_loaders, model, writer)
    print("Used run_code: {}".format(args.run_code))
