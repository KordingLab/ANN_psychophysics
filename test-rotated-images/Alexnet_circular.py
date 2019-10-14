import argparse
import os
import random
import shutil
import time
import warnings
import pickle


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.nn.modules.conv import _ConvNd
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple


def make_circle(mask):
    """From a 2d array of all zeros we make the center ellipse all 1s, zeros outside"""
    s = mask.size()
    assert len(s)==2
    for row in torch.arange(0,s[0], dtype = torch.float):
        for col in torch.arange(0,s[1],dtype = torch.float):
            dist =  torch.sqrt(((.5+row-s[0]/2)**2)/(s[0]/2)**2 + ((.5+col-s[1]/2)**2)/(s[1]/2)**2)
            if dist < 1:
                mask[int(row),int(col)] = 1
                
    return mask

class circ_Conv2d(_ConvNd):
    """Edited version of conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(circ_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        mask = torch.zeros_like(self.weight[0,0])
        self.mask = make_circle(mask)
        
    def _apply(self, fn):
        """So the mask is moved to device too."""
        super(circ_Conv2d, self)._apply(fn)
        self.mask = fn(self.mask)
        return self

    def conv2d_forward_masked(self, input, weight):
        # apply the mask. we're not freezing the weights but they'll always be zero
        self.weight.data.mul_(self.mask)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward_masked(input, self.weight)

class AlexNet_circular(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_circular, self).__init__()
        self.features = nn.Sequential(
            circ_Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            circ_Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            circ_Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            circ_Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
