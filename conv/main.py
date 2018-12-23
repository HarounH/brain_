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
    generators,
    discriminators,
    gps,
    shacgan,
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
    parser.add_argument("--image_dir", dest="image_dir", type=str, default="images", help="Subdirectory to save images")  # noqa

    # training
    parser.add_argument("-nb", "--nbatches", dest="nbatches", type=int, metavar='<int>', default=200000, help="Number of batches to train on")  # noqa
    parser.add_argument("-brk", "--brk", dest="to_break", type=int, metavar='<int>', default=-1, help="Number of batches after which to breakpoint")  # noqa
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")  # noqa
    parser.add_argument("-gi", "--gen_iters", dest="gen_iters", type=int, metavar='<int>', default=1, help="Number of generator iters per batch")  # noqa
    parser.add_argument("-di", "--disc_iters", dest="disc_iters", type=int, metavar='<int>', default=1, help="Number of discriminator per batch")  # noqa
    parser.add_argument("-disc_lr", "--disc_learning_rate", dest="disc_lr", type=float, metavar='<float>', default=0.0001, help='Learning rate')  # noqa
    parser.add_argument("-gen_lr", "--gen_learning_rate", dest="gen_lr", type=float, metavar='<float>', default=0.0001, help='Learning rate')  # noqa
    parser.add_argument("--gp_lambda", dest="gp_lambda", default=10.0, type=float, help="gradient penalty multiplier")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0.0, help='Weight decay')  # noqa
    parser.add_argument("-clfw", "--classification_weight", dest="clfw", default=1.0, type=float, help="Weight to use on classification loss. Tweak to avoid class dependency")

    parser.add_argument('-ve', '--validate_every', type=int, default=100, help='How often to validate')  # noqa
    parser.add_argument('-se', '--save_every', type=int, default=1000, help='How often to save during training')  # noqa
    # parser.add_argument('-pe', '--print_every', type=int, default=20, help='How often to print losses during training')  # noqa

    parser.add_argument('-gt', "--generator_type", type=str, default='0', choices=generators.versions.keys(), help="What generator version to use")
    parser.add_argument('-dt', "--discriminator_type", type=str, default='0',  choices=discriminators.versions.keys(), help="What discriminator version to use")
    parser.add_argument('-gpt', "--gp_type", type=str, choices=gps.versions.keys(), default='wgangp', help="What gradient penalty version to use")
    parser.add_argument('-bm', "--gen_mask", dest="gen_mask", default=True, action="store_false", help="Mask output from generator")
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
    args.image_dir = os.path.join(args.base_output, args.run_code, args.image_dir)  # noqa

    directories_needed = [args.tensorboard_dir, args.model_dir, args.image_dir]

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
                r_labels = torch.ones((batch_size), dtype=torch.float, device=args.device)
                f_labels = torch.zeros((batch_size), dtype=torch.float, device=args.device)
                if args.cuda:
                    x = x.cuda()
                    svec = svec.cuda()
                    tvec = tvec.cuda()
                    cvec = cvec.cuda()
                # Generator
                z = torch.randn(batch_size, model.LATENT_SIZE).float().to(args.device)
                gz = model.generator(z, svec, tvec, cvec)
                xs = [
                    ('real-', x),
                    ('fake-', gz),
                ]
                for tag, xin in xs:
                    rf, spred, tpred, cpred = model.discriminator(xin, predict_s=r_labels, svec=svec, tvec=tvec)
                    losses[tag + "dreal"].append(torch.log(1e-8 + rf).mean().item())
                    losses[tag + "dfake"].append(torch.log(1e-8 + 1 - rf).mean().item())
                    losses[tag + "study acc"].append((torch.argmax(spred.detach(), dim=1) == svec).float().mean().item())
                    losses[tag + "study ce"].append(F.cross_entropy(spred, svec).item())
                    losses[tag + "task acc"].append((torch.argmax(tpred.detach(), dim=1) == tvec).float().mean().item())
                    losses[tag + "task ce"].append(F.cross_entropy(tpred, tvec).item())
                    losses[tag + "contrast acc"].append((torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item())
                    losses[tag + "contrast ce"].append(F.cross_entropy(cpred, cvec).item())

                losses["disc loss"] = losses["real-dreal"][-1] + losses["fake-dfake"][-1]
                losses["gen loss"] = - losses["fake-dreal"][-1]

                # Save gz
                if not_dumped_images and (bidx % 10 == save_on_batch):
                    not_dumped_images = False
                    batch_dir = os.path.join(args.image_dir, "batch{}".format(batch_idx))
                    os.makedirs(batch_dir, exist_ok=True)
                    for i in range(0, gz.shape[0], 10):  # For each image in the batch
                        utils.save_images(
                            [gz[i].view(*constants.IMAGE_SHAPE)],
                            ["generated"],
                            os.path.join(batch_dir, "gen_{}_{}_{}_{}.png".format(args.meta['i2s'][svec[i].item()], args.meta['i2t'][tvec[i].item()], args.meta['i2c'][cvec[i].item()], i)),
                            indexes=[1],
                            nrows=1,
                            mu_=datasets[s].mu,
                            std_=datasets[s].std,
                        )
                # Log stuff
            writer.add_scalars(
                (prefix + "-" if prefix != "" else "") + s,
                {k: np.mean(v) for k, v in losses.items()},
                batch_idx
            )
    model = model.train()
    print("{}testing took {}s".format(prefix + " " if prefix != "" else "", time.time() - start))


def train(args, train_datasets, train_loaders, test_datasets, test_loaders, model, writer, prefix="train", checkpoint=None):
    gen_params = list(model.generator.parameters())
    disc_params = list(model.discriminator.parameters())
    disc_optimizer = optim.Adam(
        disc_params,
        lr=args.disc_lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    gen_optimizer = optim.Adam(
        gen_params,
        lr=args.gen_lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    if checkpoint:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        start_batch = load_state_dict(checkpoint['start_batch'])
    else:
        start_batch = 0
    to_break = args.to_break

    # Unused variable.
    r_labels = torch.ones((args.batch_size), dtype=torch.float, device=args.device)
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
            for didx in range(args.disc_iters):  # Number of discriminator iterations
                temp_time = time.time()
                disc_optimizer.zero_grad()
                rf, spred, tpred, cpred = model.discriminator(x, predict_s=r_labels, svec=svec, tvec=tvec)
                # pdb.set_trace()
                L_gan = -torch.log(1e-8 + rf).mean()  # F.binary_cross_entropy(rf.view(-1, 1), r_labels)
                L_clf_s = F.cross_entropy(spred, svec)
                L_clf_t = F.cross_entropy(tpred, tvec)
                L_clf_c = F.cross_entropy(cpred, cvec)
                disc_real_loss = L_gan + args.clfw * (L_clf_s + L_clf_t + L_clf_c)
                acc_s = (torch.argmax(spred.detach(), dim=1) == svec).float().mean().item()
                acc_t = (torch.argmax(tpred.detach(), dim=1) == tvec).float().mean().item()
                acc_c = (torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item()

                forward_prop_time += time.time() - temp_time
                temp_time = time.time()

                disc_real_loss.backward()

                backward_prop_time += time.time() - temp_time
                losses["real-" + "dreal"] = L_gan.item()
                losses["real-" + "s acc"] = acc_s
                losses["real-" + "s ce"] = L_clf_s.item()
                losses["real-" + "t acc"] = acc_t
                losses["real-" + "t ce"] = L_clf_t.item()
                losses["real-" + "c acc"] = acc_c
                losses["real-" + "c ce"] = L_clf_c.item()

                z = torch.randn(batch_size, model.LATENT_SIZE, device=args.device, dtype=torch.float)
                Gz = model.generator(z, svec, tvec, cvec)
                rf, spred, tpred, cpred = model.discriminator(x)  # , predict_s=r_labels, svec=svec, tvec=tvec)
                # pdb.set_trace()
                L_gan = -torch.log(1e-8 + 1 - rf).mean()  # F.binary_cross_entropy(rf.view(-1, 1), f_labels)
                # L_clf_s = F.cross_entropy(spred, svec)
                # L_clf_t = F.cross_entropy(tpred, tvec)
                # L_clf_c = F.cross_entropy(cpred, cvec)
                disc_fake_loss = L_gan  # + L_clf_s + L_clf_t + L_clf_c
                losses["fake-" + "dfake"] = L_gan.item()

                losses["disc loss no gp"] = disc_real_loss.item() + disc_fake_loss.item()

                forward_prop_time += time.time() - temp_time
                temp_time = time.time()

                disc_fake_loss.backward()

                backward_prop_time += time.time() - temp_time
                grad_start = time.time()
                # gradient_penalty = get_gradient_penality(args, model, x, Gz)
                gradient_penalty = model.gradient_penalty(model.discriminator, x, Gz)
                losses["gradient_penalty"] = gradient_penalty.item()
                gradient_penalty.backward()
                gradient_penalty_time += time.time() - grad_start
                # Compute gradient norms
                for pn, p in model.discriminator.named_parameters():
                    if p.requires_grad and not("bias" in pn):
                        losses["norm-disc-{}".format(pn)] = p.detach().norm().item()
                        losses["norm-disc-grad-{}".format(pn)] = p.grad.detach().norm().item()
                        # if torch.isnan(p.grad).any():
                        #     pdb.set_trace()
                disc_optimizer.step()

            for i in range(args.gen_iters):
                ####################
                # GENERATOR
                ####################
                gen_optimizer.zero_grad()
                temp_time = time.time()
                z = torch.randn(batch_size, model.LATENT_SIZE, device=args.device, dtype=torch.float)
                Gz = model.generator(z, svec, tvec, cvec)

                rf, spred, tpred, cpred = model.discriminator(Gz, predict_s=r_labels, svec=svec, tvec=tvec)
                L_gan = - torch.log(1e-8 + rf).mean()  # Non saturating version of minimax loss
                L_clf_s = F.cross_entropy(spred, svec)
                L_clf_t = F.cross_entropy(tpred, tvec)
                L_clf_c = F.cross_entropy(cpred, cvec)
                gen_loss = L_gan + args.clfw * (L_clf_s + L_clf_t + L_clf_c)
                acc_s = (torch.argmax(spred.detach(), dim=1) == svec).float().mean().item()
                acc_t = (torch.argmax(tpred.detach(), dim=1) == tvec).float().mean().item()
                acc_c = (torch.argmax(cpred.detach(), dim=1) == cvec).float().mean().item()

                losses["fake-" + "dreal"] = L_gan.item()
                losses["fake-" + "s acc"] = acc_s
                losses["fake-" + "s ce"] = L_clf_s.item()
                losses["fake-" + "t acc"] = acc_t
                losses["fake-" + "t ce"] = L_clf_t.item()
                losses["fake-" + "c acc"] = acc_c
                losses["fake-" + "c ce"] = L_clf_c.item()
                losses["gen loss"] = gen_loss.item()

                forward_prop_time += time.time() - temp_time
                temp_time = time.time()

                gen_loss.backward()
                for pn, p in model.generator.named_parameters():
                    if p.requires_grad and not("bias" in pn):
                        losses["norm-gen-{}".format(pn)] = p.detach().norm()
                        losses["norm-gen-grad-{}".format(pn)] = p.grad.detach().norm()
                        # if torch.isnan(p.grad).any():
                        #     pdb.set_trace()
                gen_optimizer.step()

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
        print("[{} / {}]: writing={:.2f} cuda={:.2f} forw={:.2f} back={:.2f} gp={:.2f} total={:.2f}".format(batch_idx, args.nbatches, writing_time, cuda_transfer_time, forward_prop_time, backward_prop_time, gradient_penalty_time, time.time() - start))
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
                model = model.toggle_dataparallel()  # Potentially slow.
                checkpoint['model'] = model.state_dict()
                model = model.toggle_dataparallel()
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
    model = shacgan.SHACGAN(
        args,
        gen_mask=constants.brain_mask_tensor if args.gen_mask else None,
        gen_version=args.generator_type,
        disc_version=args.discriminator_type,
        gradient_penalty=args.gp_type,
        loadable_state_dict=model_state_dict
    )
    if args.cuda:
        model = model.cuda()
    if args.dataparallel:
        model = model.toggle_dataparallel()

    # dump argparse
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    utils.dump_everything(args)

    # train/test
    if args.mode == 'train':
        train(args, trn, train_loaders, tst, test_loaders, model, writer, prefix="train", checkpoint=checkpoint)
    elif args.mode == 'test':
        test(args, tst, test_loaders, model, writer)
    print("Used run_code: {}".format(args.run_code))
