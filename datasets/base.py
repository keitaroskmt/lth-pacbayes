import os
import pathlib

import torch
import torchvision
import numpy as np


class CIFAR10(torchvision.datasets.CIFAR10):
    num_class = 10
    seed = 0
    def __init__(self, random_labels_fraction=None, **kwargs):
        super().__init__(**kwargs)
        if random_labels_fraction is not None:
            self.randomize_label(random_labels_fraction)

    def randomize_label(self, fraction):
        labels = np.array(self.targets)
        num_to_randomize = np.ceil(len(labels) * fraction).astype(int)
        randomized_labels = np.random.RandomState(seed=self.seed).randint(self.num_class, size=num_to_randomize)
        examples_to_randomize = np.random.RandomState(seed=self.seed+1).permutation(len(labels))[:num_to_randomize]
        labels[examples_to_randomize] = randomized_labels
        self.targets = labels


def get_dataset(dataset_name: str, random_labels_fraction = None):
    if dataset_name == 'MNIST':
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()] +
            [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        )
        train_set = torchvision.datasets.MNIST(
            train=True,
            root=os.path.join(os.getcwd(), 'data/mnist'),
            download=True,
            transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            train=False,
            root=os.path.join(os.getcwd(), 'data/mnist'),
            download=True,
            transform=transform
        )
        return train_set, test_set

    elif dataset_name == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_set = CIFAR10(
            random_labels_fraction=random_labels_fraction,
            train=True,
            root=os.path.join(os.getcwd(), 'data/cifar10'),
            download=True,
            transform=transform_train
        )
        test_set = CIFAR10(
            train=False,
            root=os.path.join(os.getcwd(), 'data/cifar10'),
            download=True,
            transform=transform_test
        )
        return train_set, test_set

    else:
        raise NotImplementedError()


def get_dataloader(train_set, test_set, batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle=False
    )

    return train_loader, test_loader
