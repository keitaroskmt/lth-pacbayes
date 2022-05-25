import numpy as np

from models.base import Model
from models.mask import Mask
from pruning import base


class Strategy(base.Strategy):
    @staticmethod
    def sparse_global(model: Model, current_mask: Mask, args) -> Mask:
        current_mask = current_mask.numpy()

        # Determine the number of weights that need to be pruned
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(args.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned
        prunable_tensors = set(model.prunable_layer_names)
        if args.pruning_layers_to_ignore:
            prunable_tensors -= set(args.pruning_layers_to_ignore.split(','))

        # Get the model weights
        weights = {k: v.clone().cpu().detach().numpy()
                    for k, v in model.state_dict().items() if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        thereshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        rand_mask = np.random.binomial(1, 0.8, np.shape(weights['conv1.weight']))

        new_mask = Mask({k: np.where(current_mask[k] == 0, current_mask[k], np.random.binomial(1, 0.8, np.shape(v))) for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
