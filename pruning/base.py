import abc

from models.base import Model
from models.mask import Mask


class Strategy(abc.ABC):
    @staticmethod
    @abc.abstractclassmethod
    def prune(model: Model, mask: Mask):
        pass
