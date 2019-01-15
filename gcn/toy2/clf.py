import argparse
import pdb
import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gcn.modules.fgl as fgl
import gcn.toy2.dataset as dset
import torch.optim as optim
import scipy.sparse as sp
import utils.utils as utils


class FGLNet(nn.Module):
    def __init__(self, midc, adj_list, *args, **kwargs):
        super(FGLNet, self).__init__()
        self.net = nn.Sequential(
            fgl.FGL_v1(3, 784, midc, len(adj_list), adj_list),
            nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Linear(len(adj_list) * midc, dset.ydim[0]),
        )  # Softmax on top

    def forward(self, x):
        cur = x  # x: N, 784, 3
        cur = self.net(cur).view(x.shape[0], -1)
        return self.linear(cur)


class Baseline(nn.Module):
    def __init__(self, cs, *args, **kwargs):
        super(Baseline, self).__init__()
        self.cs = cs
        self.net = []
        for i in range(0, len(cs) - 1):
            self.net.extend([nn.Linear(cs[i], cs[i+1]), nn.Tanh()])
        self.net.extend([nn.Linear(cs[-1], 8)])
        self.net = nn.Sequential(*(self.net))

    def forward(self, x):
        # x: N * xdim[0], xdim[1]
        cur = x.view(x.shape[0], -1)
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for module in self.net:
            cur = module(cur)
        # print(prof)
        return cur  # N, 8


arch_dict = {
    'baseline': lambda ks, As: Baseline(ks),
    'fgl': lambda ks, As: FGLNet(ks, As),
}


def test(model, test_loader, prefix):
    # model = model.eval()
    with torch.no_grad():
        losses = []
        accs = []
        start = time.time()
        for x, y in test_loader:
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            yhat = model(x).view(y.shape[0], -1)
            loss = F.cross_entropy(yhat, y).item()
            acc = (torch.argmax(yhat.detach(), dim=1) == y).float().mean().item()
            accs.append(acc)
            losses.append(loss)
        print("[{}: {}s] loss={} acc={}".format(prefix, time.time() - start, np.mean(losses), np.mean(accs)))
    # model = model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="Where is the data")
    parser.add_argument("model_type", choices=arch_dict.keys(), help="What model to use")
    parser.add_argument("-o", "--output", default="", type=str, help="Output file name. If not given, will create one")

    parser.add_argument("-ic", "--intermediate_channel", type=int, default=1, help="Number of channels in intermediate layer")
    parser.add_argument("-ad", "--density_A", type=float, default=0.01, help="Density of intermediate connection")

    parser.add_argument("-sf", "--split_fraction", type=float, default=0.8, help="Fraction of data to use for training")
    parser.add_argument("-s", "--seed", type=int, default=1337, help="Seed for dataset generation")

    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
    parser.add_argument("-dp", "--dataparallel", dest="dataparallel", default=False, action="store_true")  # noqa

    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs of training")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0.0, help='Weight decay')  # noqa
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="Seed for dataset generation")
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.dataparallel = args.dataparallel and args.cuda

    if args.output == "":
        args.output = os.path.join("gcn/toy2/outputs", args.model_type, "ic{}dA{}s{}".format(args.intermediate_channel, args.density_A, args.seed), os.path.basename(args.datafile).replace(".npz", ".chk"))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Writing checkpoint to {}".format(args.output))

    train_dset, test_dset = dset.load_train_test(args.datafile, args.split_fraction)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    # Construct As
    if args.model_type == 'baseline':
        As = None
        ks = [np.prod(dset.xdim), args.intermediate_channel * np.prod(dset.hdim), dset.ydim[0]]
    else:
        As = []
        base_As = train_dset.Ahx
        for ridx in range(base_As.shape[0]):
            As.append([ridx])
            for cidx in range(base_As.shape[1]):
                if base_As[ridx, cidx] > 0:
                    As[-1].append(cidx)
        ks = args.intermediate_channel

    model = arch_dict[args.model_type](ks, As)  # Pass in other things too.

    if args.dataparallel:
        model = nn.DataParallel(model.cuda())
    elif args.cuda:
        # print("Calling CUDA")
        model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay

    )
    # Training
    for eidx in range(args.epochs):
        losses = []
        accs = []
        start = time.time()
        for x, y in train_loader:
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            # pdb.set_trace()
            yhat = model(x).view(y.shape[0], -1)
            loss = F.cross_entropy(yhat, y)

            acc = (torch.argmax(yhat.detach(), dim=1) == y).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accs.append(acc)
            losses.append(loss.item())
        print("[{}/{}: {}s] loss={} acc={}".format(eidx, args.epochs, time.time() - start, np.mean(losses), np.mean(accs)))
        if eidx % 1 == 0:
            test(model, test_loader, "val {}/{}".format(eidx, args.epochs))
    checkpoint = {'model': model.state_dict(), 'args': args.__dict__}
    torch.save(
        checkpoint,
        args.output
    )
    del checkpoint
    # Testing
    test(model, test_loader, "test")
