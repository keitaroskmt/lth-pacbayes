from models.base import Model
from models.mask import Mask


class PrunedModel(Model):
    def __init__(self, model: Model, mask: Mask):
        if isinstance(model, PrunedModel):
            raise ValueError('Cannot nest pruned models')
        super(PrunedModel, self).__init__()
        self.model = model

        for k, v in mask.items():
            self.register_buffer(PrunedModel.to_mask_name(k), v)
        self.apply_mask()

    def apply_mask(self):
        for name, param in self.model.named_parameters():
            mask_name = PrunedModel.to_mask_name(name)
            if hasattr(self, mask_name):
                param.data *= getattr(self, mask_name)

    def forward(self, x):
        self.apply_mask()
        return self.model.forward(x)

    def apply_mask_to_grad(self):
        '''
        Apply the mask to the gradient of the model
        '''
        for name, param in self.model.named_parameters():
            mask_name = PrunedModel.to_mask_name(name)
            if hasattr(self, mask_name):
                param.grad *= getattr(self, mask_name)

    @staticmethod
    def initialize(w):
        raise NotImplementedError()

    @staticmethod
    def to_mask_name(name):
        '''
        Return the mask name because the buffer name cannot contain '.'
        '''
        return 'mask_' + name.replace('.', '_')
