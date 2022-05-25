import abc
import torch.nn as nn

class Model(abc.ABC, nn.Module):
    @property
    def prunable_layer_names(self):
        '''
        Only the convolutional layers and fully-connected layers are pruned, and bias layers are not pruned.
        '''
        return [name + '.weight' for name, module in self.named_modules() if
            isinstance(module, nn.modules.conv.Conv2d) or
            isinstance(module, nn.modules.linear.Linear)]

    @staticmethod
    @abc.abstractmethod
    def initialize(self):
        pass
