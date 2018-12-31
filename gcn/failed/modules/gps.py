import os
import json
import argparse
import numpy as np
import scipy
import random
from random import shuffle
from time import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import torch
from torch import autograd
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from tensorboardX import SummaryWriter
from torch.nn.utils import weight_norm, spectral_norm
from torch.utils.data.dataset import Dataset


class DRAGAN(nn.Module):
    def __init__(self, lmbda=10.0):
        super(DRAGAN, self).__init__()
        self.lmbda = lmbda

    def forward(self, disc, x, *args, **kwargs):
        bs = x.shape[0]
        alpha = torch.rand(bs, 1, 1).to(x.device).expand(x.shape)

        interp = alpha * x.detach() + (1 - alpha) * (x.detach() + 0.5 * x.detach().std() * torch.rand(x.shape, device=x.device))
        interp.requires_grad_(True)
        out = disc(interp)

        if isinstance(out, tuple) or isinstance(out, list):
            dinterp = out[0]  # The others don't matter.
        else:
            dinterp = out
        gradients = autograd.grad(
            outputs=dinterp, inputs=interp,
            grad_outputs=torch.ones(dinterp.size(), device=x.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = self.lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class WGANGP(nn.Module):
    def __init__(self, lmbda=10.0):
        super(WGANGP, self).__init__()
        self.lmbda = lmbda

    def forward(self, disc, x, Gz, *args, **kwargs):
        # x, Gz have shape N, n, 1
        bs = x.shape[0]
        alpha = torch.rand(bs, 1, 1).to(x.device).expand(x.shape)
        interp = alpha * x.detach() + (1 - alpha) * Gz.detach()
        interp.requires_grad_(True)

        out = disc(interp)

        if isinstance(out, tuple) or isinstance(out, list):
            dinterp = out[0]  # The others don't matter.
        else:
            dinterp = out

        gradients = autograd.grad(
            outputs=dinterp, inputs=interp,
            grad_outputs=torch.ones(dinterp.size(), device=x.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = self.lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


versions = {
    'dragan': DRAGAN,
    'wgangp': WGANGP,
}
