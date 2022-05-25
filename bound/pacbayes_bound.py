import torch
import torch.nn.functional as F
import math

from bound.perturbed_model import PerturbedLayer


class PacBayesBound():
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def calc_risk(self, loader, model=None):
        correct = 0
        total = 0
        for examples, labels in loader:
            examples = examples.to(self.device)
            labels = labels.to(self.device)
            output = self.net(examples) if model is None else model(examples)

            correct += torch.sum(torch.eq(labels, output.argmax(dim=1)))
            total += len(labels)

        return 1 - correct / total

    def calc_surrogate(self, x, y):
        output = self.net(x)
        output = F.softmax(output, dim=-1)
        y = y.view([y.shape[0], -1])
        log_likelihood = output.gather(1, y).clamp(min=1e-4, max=1).log()
        surrogate = -log_likelihood
        surrogate = surrogate.div(math.log(output.shape[-1]))
        return surrogate.mean()

    def calc_kl(self):
        kl = 0
        for layer in self.net.modules():
            if isinstance(layer, PerturbedLayer):
                kl += layer.kl_oracle_gaussian()
        return kl

    @staticmethod
    def quad_bound(risk, kl, dataset_size, delta):
        B = kl.add(math.log(2 * math.sqrt(dataset_size) / delta)).div(dataset_size)
        return risk.add(B.add(B.mul(B.add(risk.mul(2))).sqrt()))

    @staticmethod
    def pinsker_bound(risk, kl, dataset_size, delta):
        B = kl.add(math.log(2 * math.sqrt(dataset_size) / delta)).div(dataset_size)
        return risk.add(B.div(2).sqrt())

    def calc_bound(self, x, y, dataset_size, delta, n_exec=1):
        risk = 0
        for _ in range(n_exec):
            risk += self.calc_surrogate(x, y)
        risk /= n_exec
        kl = self.calc_kl()
        return torch.min(
            PacBayesBound.quad_bound(risk, kl, dataset_size, delta),
            PacBayesBound.pinsker_bound(risk, kl, dataset_size, delta)
        )

    def calc_bound_loader(self, loader, dataset_size, delta, n_exec=10):
        risk = 0
        for _ in range(n_exec):
            risk += self.calc_risk(loader)
        risk /= n_exec
        kl = self.calc_kl()
        return (risk, kl, torch.min(
            PacBayesBound.quad_bound(risk, kl, dataset_size, delta),
            PacBayesBound.pinsker_bound(risk, kl, dataset_size, delta)
        ))


class PacBayesBound_LTH(PacBayesBound):
    def __init__(self, net, device, mask):
        super().__init__(net, device)
        self.mask = mask

    def calc_kl_spike_and_slab(self):
        kl = 0
        for layer in self.net.modules():
            if isinstance(layer, PerturbedLayer):
                kl += layer.kl_oracle_spike_and_slab()
        return kl

    def calc_kl_gaussian(self):
        kl = 0
        for layer in self.net.modules():
            if isinstance(layer, PerturbedLayer):
                kl += layer.kl_oracle_gaussian()
        return kl

    def calc_kl_gaussian_zero(self):
        kl = 0
        for layer in self.net.modules():
            if isinstance(layer, PerturbedLayer):
                kl += layer.kl_oracle_gaussian_zero()
        return kl

    def calc_bound(self, x, y, dataset_size, delta, dist_type, n_exec=1):
        risk = 0
        for _ in range(n_exec):
            risk += self.calc_surrogate(x, y)
        risk /= n_exec

        # calc kl
        if dist_type == 'spike-and-slab':
            kl = self.calc_kl_spike_and_slab()
        elif dist_type == 'gaussian':
            kl = self.calc_kl_gaussian()
        elif dist_type == 'gaussian_zero':
            kl = self.calc_kl_gaussian_zero()
        else:
            raise(ValueError('No distribution type: {}'.format(dist_type)))

        return torch.min(
            PacBayesBound.quad_bound(risk, kl, dataset_size, delta),
            PacBayesBound.pinsker_bound(risk, kl, dataset_size, delta)
        )

    def calc_bound_loader(self, loader, dataset_size, delta, dist_type, n_exec=10):
        risk = 0
        for _ in range(n_exec):
            risk += self.calc_risk(loader)
        risk /= n_exec

        # calc kl
        if dist_type == 'spike-and-slab':
            kl = self.calc_kl_spike_and_slab()
        elif dist_type == 'gaussian':
            kl = self.calc_kl_gaussian()
        elif dist_type == 'gaussian_zero':
            kl = self.calc_kl_gaussian_zero()
        else:
            raise(ValueError('No distribution type: {}'.format(dist_type)))

        return (risk, kl, torch.min(
            PacBayesBound.quad_bound(risk, kl, dataset_size, delta),
            PacBayesBound.pinsker_bound(risk, kl, dataset_size, delta)
        ))

    def calc_expected_sharpness(self, loader, model, n_exec=30):
        risk = 0
        for _ in range(n_exec):
            risk += self.calc_risk(loader)
        risk /= n_exec

        return risk - self.calc_risk(loader, model)
