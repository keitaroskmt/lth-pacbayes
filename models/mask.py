from __future__ import annotations
import os
import torch
import numpy as np

from models.base import Model

class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict:
            for k, v in other_dict.items():
                self[k] = v

    def __setitem__(self, k, v) -> None:
        if isinstance(v, np.ndarray):
            v = torch.as_tensor(v)
        super(Mask, self).__setitem__(k, v)

    @staticmethod
    def ones_like(model: Model) -> Mask:
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def to_device(self, device) -> Mask:
        return Mask({k: v.to(device) for k, v in self.items()})

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    def save(self, output_path):
        torch.save({k: v.cpu().int() for k, v in self.items()}, output_path)

    @staticmethod
    def load(output_path):
        if not os.path.exists(output_path):
            raise ValueError()
        return torch.load(output_path)

    @property
    def sparsity(self):
        '''
        Return the percent of weights that have been pruned in the prunable layers
        '''
        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity
