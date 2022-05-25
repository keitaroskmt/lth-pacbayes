import torch
import torch.nn as nn

from models import base

class Model(base.Model):
    def __init__(self, n_classes=10):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def initialize(w):
        # Initialize weights
        if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
            nn.init.kaiming_normal_(w.weight)

        # Initialize the Batch Normalization layers
        if isinstance(w, nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)
