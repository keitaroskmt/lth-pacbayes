import argparse
import sys
import os
import pathlib
import json

import torch

current_dir = os.path.join(pathlib.Path().resolve())
sys.path.append(str(current_dir))

from models.util import get_model
from models.mask import Mask
from bound import perturbed_mnist_lenet, perturbed_cifar_resnet
from bound.pacbayes_bound import PacBayesBound_LTH
from datasets.base import get_dataset, get_dataloader


def main(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    output_path = os.path.join(os.getcwd(), "lottery_data", args.hash_name, "replicate_{}".format(args.replicate))

    init_state_dict = torch.load(os.path.join(output_path, "level_{}".format(args.level), "init_model.pth"), map_location=device)
    trained_state_dict = torch.load(os.path.join(output_path, "level_{}".format(args.level), "trained_model.pth"), map_location=device)
    mask = Mask(torch.load(os.path.join(output_path, "level_{}".format(args.level), "mask.pth"), map_location=device))
    mask = mask.to_device(device)

    prior_mean = get_model(args.model_name, args.dataset_name, device)
    prior_mean.load_state_dict(init_state_dict)
    posterior_mean = get_model(args.model_name, args.dataset_name, device)
    posterior_mean.load_state_dict(trained_state_dict)

    is_spike_and_slab = (args.dist_type == 'spike-and-slab')
    if args.model_name == 'lenet':
        perturbed_model = perturbed_mnist_lenet.Model(prior_mean=prior_mean, posterior_mean=posterior_mean, mask=mask, init_var=args.init_var, target_sparsity=mask.sparsity, mask_noise=is_spike_and_slab)
    else:
        perturbed_model = perturbed_cifar_resnet.Model(prior_mean=prior_mean, posterior_mean=posterior_mean, mask=mask, init_var=args.init_var, target_sparsity=mask.sparsity, mask_noise=is_spike_and_slab)

    perturbed_model.to(device)
    pacbayes_bound = PacBayesBound_LTH(perturbed_model, device, mask)

    # dataset
    train_set, test_set = get_dataset(args.dataset_name)
    train_loader, test_loader = get_dataloader(train_set, test_set, 256)
    train_set_size = len(train_set)

    # training configuration
    optimizer = torch.optim.SGD(perturbed_model.parameters(), lr=args.lr, momentum=0.95)

    bound_list = []
    risk_list = []
    kl_list = []
    test_loss_list = []
    for ep in range(args.epoch):
        perturbed_model.train()
        for examples, labels in train_loader:
            examples = examples.to(device)
            labels = labels.to(device)
            bound = pacbayes_bound.calc_bound(examples, labels, train_set_size, args.delta, args.dist_type)
            optimizer.zero_grad()
            bound.backward()
            optimizer.step()
            print("Bound: {}".format(bound))

        perturbed_model.eval()
        with torch.no_grad():
            risk, kl, bound = pacbayes_bound.calc_bound_loader(train_loader, train_set_size, args.delta, args.dist_type)
            test_loss = pacbayes_bound.calc_risk(test_loader)
            print("Dist: {}, Level: {}, Epoch: {}, Bound: {}, Test Loss: {}".format(args.dist_type, args.level, ep, bound, test_loss))
            bound_list.append(bound.item())
            risk_list.append(risk.item())
            kl_list.append(kl.item())
            test_loss_list.append(test_loss.item())

    with open(os.path.join(output_path, 'level_{}'.format(args.level),'result_{}_{}.json'.format(args.dist_type, args.result_file)), 'w') as f:
        json.dump({
            "bound_list": bound_list,
            "risk_list": risk_list,
            "kl_list": kl_list,
            "test_loss_list": test_loss_list,
            "args": vars(args),
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default=10, type=int, help="Specify pruning level")
    parser.add_argument("--replicate", default=0, type=int, help="")
    parser.add_argument("--model_name", default='lenet', type=str, help="Model name (ex. lenet, resnet20)")
    parser.add_argument("--dataset_name", default='MNIST', type=str, help="Dataset name (ex. MNIST, CIFAR10)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device you want to use")

    parser.add_argument("--hash_name", default='9eb5e716e5b9f7c0ddc4251af58f6789', type=str, help="Hash name of the experiment")
    parser.add_argument("--dist_type", default='spike-and-slab', type=str, help="Distribution used in pac-bayes (ex. spike-and-slab, gaussian")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--delta", default=0.01, type=float, help="Delta in pac-bayes bound")
    parser.add_argument("--init_var", default=-4.0, type=float, help="Standard deviation applied the inverse softplus function")
    parser.add_argument("--epoch", default=256, type=int, help="Epoch")
    parser.add_argument("--result_file", default=0, type=int, help="")

    args = parser.parse_args()

    main(args)
