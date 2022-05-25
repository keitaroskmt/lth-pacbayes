import argparse
import os
import pathlib
import json

import torch
import numpy as np

from models.util import get_model
from models.mask import Mask
from models.pruned_model import PrunedModel


def main(args):
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    replicate = 0
    levels = 15

    l2_dist_list = []
    l2_norm_list = []
    l2_dist_normalized_list = []
    l2_norm_normalized_list = []
    for level in range(levels+1):
        output_path = os.path.join(os.getcwd(), "lottery_data", args.hash_name, "replicate_{}".format(replicate))
        mask = Mask(torch.load(os.path.join(output_path, "level_{}".format(level), "mask.pth")))

        init_state_dict = torch.load(os.path.join(output_path, "level_{}".format(level), "init_model.pth"))
        model_1 = get_model(args.model_name, args.dataset_name, device)
        model_1.load_state_dict(init_state_dict)
        pruned_model_1 = PrunedModel(model_1, mask)

        trained_state_dict = torch.load(os.path.join(output_path, "level_{}".format(level), "trained_model.pth"))
        model_2 = get_model(args.model_name, args.dataset_name, device)
        model_2.load_state_dict(trained_state_dict)
        pruned_model_2 = PrunedModel(model_2, mask)

        l2_dist_sum = 0
        l2_norm_sum = 0
        non_zero_sum = 0
        for (k1, v1), (k2, v2) in zip(pruned_model_1.named_parameters(), pruned_model_2.named_parameters()):
            assert(k1 == k2)
            if "conv" in k1  or "fc" in k1 or "bn" in k1: # perturbed layers
                l2_dist_sum += (v1 - v2).pow(2).sum().item()
                l2_norm_sum += v2.pow(2).sum().item()
                non_zero_sum += torch.count_nonzero(v1).item()

        l2_dist = np.sqrt(l2_dist_sum)
        l2_norm = np.sqrt(l2_norm_sum)
        l2_dist_normalized = l2_dist / np.sqrt(non_zero_sum)
        l2_norm_normalized = l2_norm / np.sqrt(non_zero_sum)

        l2_dist_list.append(l2_dist)
        l2_norm_list.append(l2_norm)
        l2_dist_normalized_list.append(l2_dist_normalized)
        l2_norm_normalized_list.append(l2_norm_normalized)

    with open(os.path.join(output_path, "distance.json"), 'w') as f:
        json.dump({
            "l2_dist": l2_dist_list,
            "l2_norm": l2_norm_list,
            "l2_dist_normalized": l2_dist_normalized_list,
            "l2_norm_normalized": l2_norm_normalized_list,
            "levels": [i for i in range(levels+1)],
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='resnet20', type=str, help="Model name (ex. lenet, resnet20)")
    parser.add_argument("--dataset_name", default='CIFAR10', type=str, help="Dataset name (ex. MNIST, CIFAR10)")
    parser.add_argument("--hash_name", default=None, type=str, help="Dataset name (ex. MNIST, CIFAR10)")

    args = parser.parse_args()

    main(args)
