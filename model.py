"""
Model for Mineral Classifier

Author: Ivan Klevanski

"""

import os
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F

from typing import Union

abs_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Default: 224 x 224 inputs

Model architecture subject to change
"""

class DSConv2d(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 kernel_size: int | tuple[int, int], 
                 stride: int | tuple[int, int] = 1, 
                 padding: Union[str, int | tuple[int, int]] = 0,
                 padding_mode: str = "zeros",
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = False):
        super(DSConv2d, self).__init__()

        self.dconv = nn.Conv2d(in_channels=in_channels, 
                               out_channels=in_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               padding_mode=padding_mode, 
                               dilation=dilation, 
                               groups=in_channels, 
                               bias=bias)
        self.pconv = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=1,
                               bias=bias)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x
    

class MineralCNNet(nn.Module):
    def __init__(self, img_dim = 224, cdim = 3, k_sizes: tuple[int] = (4, 2), p_sizes: tuple[int] = (8, 8), drop_rate: float = 0.1, num_classes = 10): # tuple[int] = (4, 4, 4), p_sizes: tuple[int] = (4, 4, 4)
            super(MineralCNNet, self).__init__()

            conv2fc_dim = img_dim

            for k, p in zip(k_sizes, p_sizes):
                conv2fc_dim -= (k - 1) * 2 # Double convolutions
                conv2fc_dim = int(conv2fc_dim / p)

            conv2fc_dim = (conv2fc_dim ** 2) * 64

            #conv2fc_dim = ((img_dim - sum(k_sizes) + len(k_sizes)) ** 2) * 96

            self.conv = nn.Sequential(
                DSConv2d(in_channels=cdim, out_channels=32, kernel_size=k_sizes[0]),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                DSConv2d(in_channels=32, out_channels=32, kernel_size=k_sizes[0]),
                nn.MaxPool2d(p_sizes[0]),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(drop_rate),
                DSConv2d(in_channels=32, out_channels=64, kernel_size=k_sizes[1]),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                DSConv2d(in_channels=64, out_channels=64, kernel_size=k_sizes[1]),
                nn.MaxPool2d(p_sizes[1]),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(drop_rate)
                
            )

            """
                DSConv2d(in_channels=64, out_channels=128, kernel_size=k_sizes[2]),
                nn.MaxPool2d(p_sizes[2]),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            """

            self.flatten = nn.Flatten()

            self.fc = nn.Sequential(
                nn.BatchNorm1d(conv2fc_dim),
                nn.Linear(in_features=conv2fc_dim, out_features=256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(in_features=64, out_features=num_classes)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
        
    def predict(self, x):
        return F.softmax(self.forward(x), dim=0)


class CNNetWrapper(nn.Module):
    def __init__(self, base_network: nn.Module, base_num_classes: int = 1000, num_classes = 10):
        super(CNNetWrapper, self).__init__()

        self.out_channels_base = base_num_classes
        self.base = base_network
        self.fc = nn.Linear(in_features=base_num_classes, out_features=num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        
        return x
        
    def predict(self, x):
        return F.softmax(self.forward(x))

"""
class MineralCNNet(nn.Module):
    def __init__(self, img_dim = 224, cdim = 3, k_sizes: tuple[int] = (4, 4, 4, 4), p_sizes: tuple[int] = (4, 2, 2, 2), drop_rate: float = 0.15, num_classes = 10):
            super(MineralCNNet, self).__init__()

            conv2fc_dim = img_dim

            for k, p in zip(k_sizes, p_sizes):
                conv2fc_dim -= k - 1
                conv2fc_dim = int(conv2fc_dim / p)

            conv2fc_dim = (conv2fc_dim ** 2) * 64

            #conv2fc_dim = ((img_dim - sum(k_sizes) + len(k_sizes)) ** 2) * 96

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=cdim, out_channels=32, kernel_size=k_sizes[0]),
                nn.MaxPool2d(p_sizes[0]),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k_sizes[1]),
                nn.MaxPool2d(p_sizes[1]),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k_sizes[2]),
                nn.MaxPool2d(p_sizes[2]),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=k_sizes[3]),
                nn.MaxPool2d(p_sizes[3]),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(drop_rate)
            )

            self.flatten = nn.Flatten()

            self.fc = nn.Sequential(
                nn.Linear(in_features=conv2fc_dim, out_features=256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(in_features=64, out_features=num_classes)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
        
    def predict(self, x):
        return F.softmax(self.forward(x))
"""
