from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
from torch import int64, Tensor
import collections
import operator
from functools import reduce
from collections import OrderedDict
from itertools import repeat
from torch.nn.functional import log_softmax
# MNIST: 1*32*32
# CIFAR10: 3*32*32

class LeNet(torch.jit.ScriptModule):
    """从李沐那里抄的 https://zh-v2.d2l.ai/chapter_convolutional-neural-networks/lenet.html
    如果输入是MNIST的1*28*28的话底下576要改成16*5*5=400"""
    def __init__(self, in_channel, out_channel):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=5, padding=2), nn.Sigmoid(), # 6 * 32 * 32
            nn.AvgPool2d(2, 2), # 6 * 16 * 16
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), # 16 * 12 * 12
            nn.AvgPool2d(kernel_size=2, stride=2), # 16 * 6 * 6 = 576
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, out_channel)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.classifier(out)
        return out

def LeNet_CIFAR10(): return LeNet(3, 10)

def ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        x = tuple(x)
        if len(x) != 1:
            assert len(x) == n, "Expected length %d, but found %d" % (n, len(x))
            return x
        else:
            x = x[0]
    return tuple(repeat(x, n))

class CNN(nn.Module):
    def __init__(self, conv_depth, num_filters, filter_size, padding_size, pool_size, dense_depth, num_dense_units, input_size, output_size, pool_every=1, batch_norm=True, dropout=0):
        super().__init__()

        # Need to keep track of dimensions to know which linear layer we need later
        in_channels = int(input_size[0])
        dims = torch.tensor(tuple(input_size[1:]), dtype=int64)

        # Expand definitions
        pool_size = ntuple(pool_size, 2)
        filter_size = ntuple(filter_size, 2)
        padding_size = ntuple(padding_size, 2)
        num_filters = ntuple(num_filters, conv_depth)
        num_dense_units = ntuple(num_dense_units, dense_depth)
        non_lin = nn.ReLU()

        # Build up convolutional layers
        conv_layers = OrderedDict()
        for i in range(conv_depth):
            layer = OrderedDict()

            layer["conv"] = nn.Conv2d(in_channels, num_filters[i], kernel_size=filter_size, padding=padding_size)
            dims -= torch.tensor(filter_size, dtype=int64) - 1 - torch.tensor(padding_size, dtype=int64) * 2
            in_channels = num_filters[i]
            # print(dims, in_channels)

            if i % pool_every == pool_every - 1:
                layer["maxpool"] = nn.MaxPool2d(pool_size)
                dims //= torch.tensor(pool_size, dtype=int64)

            if batch_norm:
                layer["batchnorm"] = nn.BatchNorm2d(num_filters[i])
            layer["nonlin"] = non_lin
            layer["dropout"] = nn.Dropout2d(dropout)

            conv_layers[f"conv_{i}"] = (nn.Sequential(layer))

        # Fully connected layers
        previous_size = in_channels * reduce(operator.mul, dims)
        dense_layers = OrderedDict()
        for i in range(dense_depth):
            layer = OrderedDict()
            layer["linear"] = nn.Linear(previous_size, num_dense_units[i])
            previous_size = num_dense_units[i]
            if batch_norm:
                layer["batch_norm"] = nn.BatchNorm1d(num_dense_units[i])
            layer["non_lin"] = non_lin
            if dropout > 0:
                layer["dropout"] = nn.Dropout(float(dropout))
            dense_layers[f"fc_{i}"] = nn.Sequential(layer)

        self.conv = nn.Sequential(conv_layers)
        self.fc = nn.Sequential(dense_layers)
        self.final = nn.Linear(previous_size, output_size)

    def forward(self, data):
        data = self.conv(data)
        data = data.reshape(data.shape[0], -1)
        data = self.fc(data)
        data = self.final(data)
        data = log_softmax(data, 1)
        return data

def CNN12_CIFAR10(): return CNN(1, 12, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN24_CIFAR10(): return CNN(1, 24, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN36_CIFAR10(): return CNN(1, 36, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN48_CIFAR10(): return CNN(1, 48, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN96_CIFAR10(): return CNN(1, 96, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN48x2_CIFAR10(): return CNN(2, 48, 5, 2, 2, 1, 256, (3, 32, 32), 10)
def CNN48x3_CIFAR10(): return CNN(3, 48, 5, 2, 2, 1, 256, (3, 32, 32), 10)