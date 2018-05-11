# Cifar10 Wideresnet for Dawn Submission

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...layers import *

def conv_2d(ni, nf, ks, stride): return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn(ni, init_zero=False):
    m = nn.BatchNorm2d(ni)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

def bn_relu_conv(ni, nf, ks, stride, init_zero=False):
    bn_initzero = bn(ni, init_zero=init_zero)
    return nn.Sequential(bn_initzero, nn.ReLU(inplace=True), conv_2d(ni, nf, ks, stride))

def noop(x): return x

class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride, drop_p=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, 3, stride)
        self.conv2 = bn_relu_conv(nf, nf, 3, 1)
        self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None
        self.shortcut = conv_2d(ni, nf, 1, stride) if ni != nf else noop

    def forward(self, x):
        x2 = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x2)
        x = self.conv1(x2)
        if self.drop: x = self.drop(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)


def _make_group(N, ni, nf, block, stride, drop_p):
    return [block(ni if i == 0 else nf, nf, stride if i == 0 else 1, drop_p) for i in range(N)]

class WideResNet(nn.Module):
    def __init__(self, num_groups, N, num_classes, k=1, drop_p=0.0, start_nf=16):
        super().__init__()
        n_channels = [start_nf]
        for i in range(num_groups): n_channels.append(start_nf*(2**i)*k)

        layers = [conv_2d(3, n_channels[0], 3, 1)]  # conv1
        for i in range(num_groups):
            layers += _make_group(N, n_channels[i], n_channels[i+1], BasicBlock, (1 if i==0 else 2), drop_p)

        layers += [nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                   Flatten(), nn.Linear(n_channels[3], num_classes)]
        self.features = nn.Sequential(*layers)

    def forward(self, x): return self.features(x)


def wrn_22(): return WideResNet(num_groups=3, N=3, num_classes=10, k=6, drop_p=0.)
def wrn_22_k8(): return WideResNet(num_groups=3, N=3, num_classes=10, k=8, drop_p=0.)
def wrn_22_k10(): return WideResNet(num_groups=3, N=3, num_classes=10, k=10, drop_p=0.)
def wrn_22_k8_p2(): return WideResNet(num_groups=3, N=3, num_classes=10, k=8, drop_p=0.2)
def wrn_28(): return WideResNet(num_groups=3, N=4, num_classes=10, k=6, drop_p=0.)
def wrn_28_k8(): return WideResNet(num_groups=3, N=4, num_classes=10, k=8, drop_p=0.)
def wrn_28_k8_p2(): return WideResNet(num_groups=3, N=4, num_classes=10, k=8, drop_p=0.2)
def wrn_28_p2(): return WideResNet(num_groups=3, N=4, num_classes=10, k=6, drop_p=0.2)

