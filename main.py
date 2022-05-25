import argparse
import os
import pathlib
import copy
import hashlib
import json

import torch
import torch.nn as nn

from models import optimizers
from models.util import get_model
from models.mask import Mask
from models.sam import enable_running_stats, disable_running_stats
from models.pruned_model import PrunedModel
from datasets.base import get_dataset, get_dataloader
from pruning.sparse_global import Strategy

def get_hash(args):
    lst = []
    for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
        if k == 'replicate' or k == 'seed' or k == 'gpu':
            continue
        lst.append(str(v))

    return hashlib.md5(";".join(lst).encode('utf-8')).hexdigest()


def main(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # dataset
    train_set, test_set = get_dataset(args.dataset_name, args.random_labels_fraction)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

    # model, dataset
    model = get_model(args.model_name, args.dataset_name, device)
    initial_state_dict = copy.deepcopy(model.state_dict())

    hash_name = get_hash(args)
    output_path = os.path.join(os.getcwd(), "lottery_data", hash_name, "replicate_{}".format(args.replicate))
    print("output path is {}".format(output_path))

    density_list = []
    best_acc_list = []
    final_acc_list = []
    mask = Mask.ones_like(model)
    mask = mask.to_device(device)
    for level in range(args.levels+1):
        if level > 0:
            mask = Strategy.sparse_global(model, mask, args)
            mask = mask.to_device(device)
        print("Level: {}, Sparsity: {}".format(level, mask.density))

        # initialize the weight
        model.load_state_dict(initial_state_dict)
        pruned_model = PrunedModel(model, mask)

        # training settings
        optimizer = optimizers.get_optimizer(pruned_model, args)
        lr_schedule = optimizers.get_lr_schedule(args, optimizer)
        criterion = nn.CrossEntropyLoss()

        # save the initial state and the pruning mask
        os.makedirs(os.path.join(output_path, 'level_{}'.format(level)), exist_ok=True)
        torch.save(pruned_model.model.to('cpu').state_dict(), os.path.join(output_path, 'level_{}'.format(level), 'init_model.pth'))
        mask.save(os.path.join(output_path, 'level_{}'.format(level), 'mask.pth'))

        # specify iteration numbers instead of epoch numbers
        iteration = 0
        ended = False
        if args.iteration is not None:
            args.epoch = args.iteration

        pruned_model.to(device)
        best_acc = 0
        for ep in range(args.epoch):
            if ended:
                break

            pruned_model.train()
            for examples, labels in train_loader:
                examples = examples.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if args.optimizer_name == 'SAM':
                    # first forward-backward step
                    enable_running_stats(pruned_model)
                    loss = criterion(pruned_model(examples), labels)
                    loss.backward()
                    pruned_model.apply_mask_to_grad()
                    optimizer.first_step(zero_grad=True)

                    # second forward-backward step
                    disable_running_stats(model)
                    loss = criterion(pruned_model(examples), labels)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    loss = criterion(pruned_model(examples), labels)
                    loss.backward()
                    optimizer.step()

                lr_schedule.step()

                iteration += 1
                if args.iteration is not None and iteration == args.iteration:
                    ended = True
                    break

            correct = 0
            total = 0
            pruned_model.eval()
            with torch.no_grad():
                for examples, labels in test_loader:
                    examples = examples.to(device)
                    labels = labels.to(device)
                    output = pruned_model(examples)

                    correct += torch.sum(torch.eq(labels, output.argmax(dim=1))).item()
                    total += len(labels)

            acc = correct / total
            print("Epoch: {}, Test Accuracy: {}".format(ep, acc))

            if acc > best_acc:
                best_acc = acc

        density_list.append(mask.density.item())
        best_acc_list.append(best_acc)
        final_acc_list.append(acc)

        # save the trained model
        torch.save(pruned_model.model.to('cpu').state_dict(), os.path.join(output_path, 'level_{}'.format(level), 'trained_model.pth'))

    with open(os.path.join(output_path, 'result.json'), 'w') as f:
        json.dump({
            "density_list": density_list,
            "best_acc_list": best_acc_list,
            "final_acc_list": final_acc_list,
            "args": vars(args)
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epoch", default=40, type=int, help="Epoch")
    parser.add_argument("--iteration", default=None, type=int, help="Iteration")
    parser.add_argument("--model_name", default='lenet', type=str, help="Model name (ex. lenet, resnet20)")
    parser.add_argument("--dataset_name", default='MNIST', type=str, help="Dataset name (ex. MNIST, CIFAR10)")
    parser.add_argument("--random_labels_fraction", default=None, type=float, help="Apply random labels to a fraction of the training set: float in (0, 1]")
    parser.add_argument("--milestone_steps", default="", type=str, help="Iterations when the learning rate is dropped")
    parser.add_argument("--gamma", default=0.1, type=float, help="How much the learning rate is dropped at each milestone step")
    parser.add_argument("--warmup_steps", default=None, type=int, help="iterations of linear the learning rate warmup")
    parser.add_argument("--optimizer_name", default='SGD', type=str, help="Optimzier name (ex. SGD, Adam, SAM")
    parser.add_argument("--weight_decay", default=None, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum")
    parser.add_argument("--rho", default=None, type=float, help="The neighberhood size to specify with SAM")
    # LTH settings
    parser.add_argument("--levels", default=15, type=int, help="Pruning levels")
    parser.add_argument("--pruning_fraction", default=0.2, type=float, help="The fraction of pruning at each iteration of iterative pruning")
    parser.add_argument("--pruning_layers_to_ignore", default=None, type=str, help="A comma-separated list of tensors that should not be pruned")
    # experiment settings
    parser.add_argument("--replicate", default=0, type=int, help="")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device you want to use")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")

    args = parser.parse_args()

    main(args)
