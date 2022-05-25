import torch
import torch.nn as nn

from bound.perturbed_model import PerturbedLinear, PerturbedConv2d
from models import base
from models.mask import Mask


class Model(nn.Module):
    def __init__(self, prior_mean: base.Model, posterior_mean: base.Model, mask: Mask = {}, init_var=-4.0, target_sparsity=None, mask_noise: bool = False):
        super(Model, self).__init__()

        conv2d_list = []
        linear_list = []
        for (name, prior_layer), (name_, posterior_layer) in zip(prior_mean.named_modules(), posterior_mean.named_modules()):
            assert(name == name_ and type(prior_layer) == type(posterior_layer))
            if isinstance(prior_layer, nn.Conv2d):
                conv2d_list.append((prior_layer, posterior_layer))
            elif isinstance(prior_layer, nn.Linear):
                linear_list.append((prior_layer, posterior_layer))

        assert(len(conv2d_list) == 2 and len(linear_list) == 3)
        self.features = nn.Sequential(
            PerturbedConv2d(*conv2d_list[0], mask.get('features.0.weight'), init_var, target_sparsity, mask_noise),
            nn.ReLU(),
            PerturbedConv2d(*conv2d_list[1], mask.get('features.2.weight'), init_var, target_sparsity, mask_noise),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            PerturbedLinear(*linear_list[0], mask.get('classifier.0.weight'), init_var, target_sparsity, mask_noise),
            nn.ReLU(),
            PerturbedLinear(*linear_list[1], mask.get('classifier.2.weight'), init_var, target_sparsity, mask_noise),
            nn.ReLU(),
            PerturbedLinear(*linear_list[2], mask.get('classifier.4.weight'), init_var, target_sparsity, mask_noise),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
