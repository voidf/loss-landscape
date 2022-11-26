from typing import Iterable, List, Tuple
import torch
import torch.nn as nn

# MNIST: 1*32*32
# CIFAR10: 3*32*32

class LeNet(nn.Module):
    def __init__(self, input_shape: Iterable[int]=(1, 32, 32)):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.classifier(out)
        return out

