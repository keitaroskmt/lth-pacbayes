import bisect
import torch
import numpy as np

from models.base import Model
from models.sam import SAM


def get_optimizer(model: Model, args):
    if args.optimizer_name == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum or 0,
            weight_decay=args.weight_decay or 0,
        )
    elif args.optimizer_name == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer_name == 'SAM':
        return SAM(
            model.parameters(),
            base_optimizer=torch.optim.SGD,
            rho=args.rho or 0.05,
            lr=args.lr,
            momentum=args.momentum or 0,
            weight_decay=args.weight_decay or 0,
        )

    raise ValueError('No such optimizer: {}'.format(args.optimizer_name))


def get_lr_schedule(args, optimzier: torch.optim.Optimizer):
    lambdas = [lambda it: 1.0]

    # drop the learning rate
    if args.gamma and args.milestone_steps:
        milestones = [int(x) for x in args.milestone_steps.split(',')]
        lambdas.append(lambda it: args.gamma ** bisect.bisect(milestones, it))

    # add linear learning rate warmup
    if args.warmup_steps:
        lambdas.append(lambda it: min(1.0, it / args.warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimzier, lambda it: np.product([l(it) for l in lambdas]))
