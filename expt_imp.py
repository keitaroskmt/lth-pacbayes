import argparse
import os
import pathlib
import json

import torch
import torch.nn as nn

from models.util import get_model
from models import optimizers
from models.mask import Mask
from models.pruned_model import PrunedModel
from datasets.base import get_dataset, get_dataloader
from pruning import base, sparse_global, sparse_global_min, sparse_global_rand


def get_acc(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(device)
            labels = labels.to(device)
            output = model(examples)

            correct += torch.sum(torch.eq(labels, output.argmax(dim=1))).item()
            total += len(labels)
    acc = correct / total
    return acc

def expt_strategy(Strategy: base.Strategy, hash_name, args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    output_path = os.path.join(os.getcwd(), "lottery_data", hash_name, "replicate_{}".format(args.replicate))

    # dataset
    train_set, test_set = get_dataset(args.dataset_name)
    train_loader, _ = get_dataloader(train_set, test_set, args.batch_size)

    trained_state_dict = torch.load(os.path.join(output_path, "level_{}".format(args.level), "trained_model.pth"))
    mask = Mask(torch.load(os.path.join(output_path, "level_{}".format(args.level), "mask.pth")))
    mask = mask.to_device(device)
    model = get_model(args.model_name, args.dataset_name, device)
    model.load_state_dict(trained_state_dict)
    new_mask = Strategy.sparse_global(model, mask, args)
    new_mask = new_mask.to_device(device)

    # accuracy after pruning
    pruned_model = PrunedModel(model, new_mask)
    pruned_model.to(device)
    acc_pruned = get_acc(pruned_model, train_loader, device)

    # accuracy after rewinding
    next_init_state_dict = torch.load(os.path.join(output_path, "level_{}".format(args.level+1), "init_model.pth"))
    model.load_state_dict(next_init_state_dict)
    pruned_model = PrunedModel(model, new_mask)
    pruned_model.to(device)
    acc_revert = get_acc(pruned_model, train_loader, device)

    # accuracy after retraining
    optimizer = optimizers.get_optimizer(pruned_model, args)
    lr_schedule = optimizers.get_lr_schedule(args, optimizer)
    criterion = nn.CrossEntropyLoss()

    iteration = 0
    ended = False
    if args.iteration is not None:
        args.epoch = args.iteration

    for _ in range(args.epoch):
        if ended:
            break

        pruned_model.train()
        for examples, labels in train_loader:
            examples = examples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            loss = criterion(pruned_model(examples), labels)
            loss.backward()
            optimizer.step()

            lr_schedule.step()
            iteration += 1
            if args.iteration is not None and iteration == args.iteration:
                ended = True
                break

    acc_retrain = get_acc(pruned_model, train_loader, device)

    return (acc_pruned, acc_revert, acc_retrain)


def expt_baseline(hash_name, args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    output_path = os.path.join(os.getcwd(), "lottery_data", hash_name, "replicate_{}".format(args.replicate))

    # dataset
    train_set, test_set = get_dataset(args.dataset_name)
    train_loader, _ = get_dataloader(train_set, test_set, args.batch_size)

    trained_state_dict = torch.load(os.path.join(output_path, "level_{}".format(args.level), "trained_model.pth"))
    mask = Mask(torch.load(os.path.join(output_path, "level_{}".format(args.level), "mask.pth")))
    mask = Mask({k: v.to(device) for k, v in mask.items()})
    model = get_model(args.model_name, args.dataset_name, device)
    model.load_state_dict(trained_state_dict)

    pruned_model = PrunedModel(model, mask)
    pruned_model.to(device)
    return get_acc(pruned_model, train_loader, device)


def main(args):
    # specify the hash names
    hash_names = []

    labels = ["IMP", "IMP_MIN", "IP_RAND"]
    accdrop_pruned_list = []
    accdrop_revert_list = []
    accdrop_retrain_list = []

    for Strategy, label in zip([sparse_global.Strategy, sparse_global_min.Strategy, sparse_global_rand.Strategy], labels):
        accdrop_pruned = 0
        accdrop_revert = 0
        accdrop_retrain = 0
        for hash_name in hash_names:
            acc_baseline = expt_baseline(hash_name, args)
            acc_pruned, acc_revert, acc_retrain = expt_strategy(Strategy, hash_name, args)
            print(hash_name, label, acc_baseline, acc_pruned, acc_revert, acc_retrain)

            accdrop_pruned += (acc_pruned - acc_baseline) / acc_baseline
            accdrop_revert += (acc_revert - acc_baseline) / acc_baseline
            accdrop_retrain += (acc_retrain - acc_baseline) / acc_baseline

        accdrop_pruned_list.append(accdrop_pruned / len(hash_names))
        accdrop_revert_list.append(accdrop_revert / len(hash_names))
        accdrop_retrain_list.append(accdrop_retrain / len(hash_names))

    with open(os.path.join(pathlib.Path.home(), "lab/pacbayes-lth/lottery_data/expt_imp.json"), 'w') as f:
        json.dump({
            "level": args.level,
            "labels": labels,
            "accdrop_pruned": accdrop_pruned_list,
            "accdrop_revert": accdrop_revert_list,
            "accdrop_retrain": accdrop_retrain_list,
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--level", default=10, type=int, help="Pruning level")
    parser.add_argument("--replicate", default=0, type=int, help="")
    parser.add_argument("--model_name", default='resnet20', type=str, help="Model name (ex. lenet, resnet20)")
    parser.add_argument("--dataset_name", default='CIFAR10', type=str, help="Dataset name (ex. MNIST, CIFAR10)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device you want to use")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    # optimize
    parser.add_argument("--lr", default=0.03, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=None, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum")
    parser.add_argument("--epoch", default=40, type=int, help="Epoch")
    parser.add_argument("--iteration", default=None, type=int, help="Iteration")
    parser.add_argument("--milestone_steps", default="", type=str, help="Iterations when the learning rate is dropped")
    parser.add_argument("--gamma", default=0.1, type=float, help="How much the learning rate is dropped at each milestone step")
    parser.add_argument("--warmup_steps", default=None, type=int, help="iterations of linear the learning rate warmup")
    parser.add_argument("--optimizer_name", default='SGD', type=str, help="Optimzier name (ex. SGD, Adam, SAM")

    # LTH
    parser.add_argument("--pruning_fraction", default=0.2, type=float, help="The fraction of pruning at each iteration of iterative pruning")
    parser.add_argument("--pruning_layers_to_ignore", default='fc.weight', type=str, help="A comma-separated list of tensors that should not be pruned")

    args = parser.parse_args()

    main(args)
