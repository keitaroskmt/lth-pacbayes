import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedLayer(nn.Module):
    def __init__(self, prior_mean: nn.Module, posterior_mean: nn.Module, mask_tensor: torch.Tensor = None, init_var=-4.0, target_sparsity=None, mask_noise=False):
        super().__init__()
        self.prior_mean = prior_mean
        self.posterior_mean = posterior_mean
        self.mask_tensor = mask_tensor
        self.init_var = init_var
        self.target_sparsity = target_sparsity
        self.mask_noise = mask_noise

        # standard deviation of weights distribution
        self.prior_var = copy.deepcopy(prior_mean)
        self.posterior_var = copy.deepcopy(posterior_mean)

        with torch.no_grad():
            for module in [self.prior_var, self.posterior_var]:
                for layer in module.parameters():
                    nn.init.constant_(layer, self.init_var)

            for module in [self.prior_mean, self.prior_var, self.posterior_mean]:
                for layer in module.parameters():
                    layer.requires_grad = False

            if mask_tensor is not None:
                for (name, p_mean), (_, q_mean) in zip(self.prior_mean.named_parameters(), self.posterior_mean.named_parameters()):
                    if name == 'weight':
                        assert(p_mean.shape == self.mask_tensor.shape and q_mean.shape == self.mask_tensor.shape)
                        q_mean.data *= self.mask_tensor

    def perturb_posterior(self):
        perturbed_weights = {}
        for (name, mean), (name_, var) in zip(self.posterior_mean.named_parameters(), self.posterior_var.named_parameters()):
            assert(name == name_)
            if self.mask_noise and name == 'weight':
                perturbed_weights[name] = mean + self.mask_tensor * (F.softplus(var) * torch.randn(mean.shape, requires_grad=False, device=mean.device))
            else:
                perturbed_weights[name] = mean + F.softplus(var) * torch.randn(mean.shape, requires_grad=False, device=mean.device)
        return perturbed_weights

    def kl_oracle_gaussian(self):
        kl = 0.0
        for p_mean, q_mean, q_var in zip(self.prior_mean.parameters(), self.posterior_mean.parameters(), self.posterior_var.parameters()):
            q_var = F.softplus(q_var).pow(2)
            kl += torch.sum(0.5 * (1 + (q_mean - p_mean).pow(2).div(q_var)).log())
        return kl

    def kl_oracle_gaussian_zero(self):
        kl = 0.0
        for q_mean, q_var in zip(self.posterior_mean.parameters(), self.posterior_var.parameters()):
            q_var = F.softplus(q_var).pow(2)
            kl += torch.sum(0.5 * (1 + (q_mean).pow(2).div(q_var)).log())
        return kl

    def kl_oracle_spike_and_slab(self):
        kl = 0.0
        for (name, p_mean), (_, q_mean), (_, q_var) in zip(
            self.prior_mean.named_parameters(), self.posterior_mean.named_parameters(), self.posterior_var.named_parameters()):
            q_var = F.softplus(q_var).pow(2)
            if name == 'weight' and self.mask_tensor is not None:
                kl += torch.where(self.mask_tensor > 0,
                    0.5 * (1 + (q_mean - p_mean).pow(2).div(q_var)).log() - (1 - self.target_sparsity).log(),
                    - self.target_sparsity.log().to(q_mean.device)
                ).sum()
            else:
                kl += torch.sum(0.5 * (1 + (q_mean - p_mean).pow(2).div(q_var)).log())
        return kl


class PerturbedLinear(PerturbedLayer):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        perturbed_weights = self.perturb_posterior()
        return F.linear(x, perturbed_weights['weight'], perturbed_weights.get('bias'))


class PerturbedConv2d(PerturbedLayer):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        perturbed_weights = self.perturb_posterior()
        return F.conv2d(x, perturbed_weights['weight'], perturbed_weights.get('bias'),
            self.posterior_mean.stride, self.posterior_mean.padding,
            self.posterior_mean.dilation, self.posterior_mean.groups)


class PerturbedBatchNorm2d(PerturbedLayer):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        perturbed_weights = self.perturb_posterior()
        return F.batch_norm(
            x, self.posterior_mean.running_mean, self.posterior_mean.running_var,
            perturbed_weights.get('weight'), perturbed_weights.get('bias'),
            self.posterior_mean.training, self.posterior_mean.momentum, self.posterior_mean.eps
        )
