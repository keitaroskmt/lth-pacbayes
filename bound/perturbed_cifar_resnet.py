import torch.nn as nn
import torch.nn.functional as F

from bound.perturbed_model import PerturbedLinear, PerturbedConv2d, PerturbedBatchNorm2d
from models import base
from models.mask import Mask


class BasicBlock(nn.Module):
    def __init__(self, layer_prior_mean, layer_posterior_mean, mask: Mask, init_var, target_sparsity, mask_noise: bool, mask_key: str):
        super().__init__()
        self.conv1 = PerturbedConv2d(layer_prior_mean.conv1, layer_posterior_mean.conv1, mask.get(mask_key + '.conv1.weight'), init_var, target_sparsity, mask_noise)
        self.bn1 = PerturbedBatchNorm2d(layer_prior_mean.bn1, layer_posterior_mean.bn1)
        self.conv2 = PerturbedConv2d(layer_prior_mean.conv2, layer_posterior_mean.conv2, mask.get(mask_key + '.conv2.weight'), init_var, target_sparsity, mask_noise)
        self.bn2 = PerturbedBatchNorm2d(layer_prior_mean.bn2, layer_posterior_mean.bn2)

        self.shortcut = layer_posterior_mean.shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, prior_mean: base.Model, posterior_mean: base.Model, mask: Mask = {}, init_var=-4.0, target_sparsity=None, mask_noise: bool = False):
        super().__init__()
        self.conv1 = PerturbedConv2d(prior_mean.conv1, posterior_mean.conv1, mask.get('conv1.weight'), init_var, target_sparsity, mask_noise)
        self.bn1 = PerturbedBatchNorm2d(prior_mean.bn1, posterior_mean.bn1)
        self.layer1 = self._make_layer(prior_mean.layer1, posterior_mean.layer1, mask, init_var, target_sparsity, mask_noise, 'layer1')
        self.layer2 = self._make_layer(prior_mean.layer2, posterior_mean.layer2, mask, init_var, target_sparsity, mask_noise, 'layer2')
        self.layer3 = self._make_layer(prior_mean.layer3, posterior_mean.layer3, mask, init_var, target_sparsity, mask_noise, 'layer3')
        self.fc = PerturbedLinear(prior_mean.fc, posterior_mean.fc, mask.get('fc.weight'), init_var, target_sparsity, mask_noise)

    def _make_layer(self, layers_prior_mean, layers_posterior_mean, mask: Mask, init_var, target_sparsity, mask_noise: bool, mask_key: str):
        layers = []
        for i, (layer_prior_mean, layer_posterior_mean) in enumerate(zip(layers_prior_mean, layers_posterior_mean)):
            layers.append(
                BasicBlock(layer_prior_mean, layer_posterior_mean, mask, init_var, target_sparsity, mask_noise, mask_key + '.{}'.format(i))
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
