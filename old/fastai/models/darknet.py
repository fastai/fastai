import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import *

class ConvBN(nn.Module):
    "convolutional layer then batchnorm"

    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x): return self.conv2(self.conv1(x)) + x

class Darknet(nn.Module):
    "Replicates the darknet classifier from the YOLOv3 paper (table 1)"

    def make_group_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in,ch_in*2,stride=stride)]
        for i in range(num_blocks): layers.append(DarknetBlock(ch_in*2))
        return layers

    def __init__(self, num_blocks, num_classes=1000, start_nf=32):
        super().__init__()
        nf = start_nf
        layers = [ConvBN(3, nf, kernel_size=3, stride=1, padding=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=(1 if i==1 else 2))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

def darknet_53(num_classes=1000):    return Darknet([1,2,8,8,4], num_classes)
def darknet_small(num_classes=1000): return Darknet([1,2,4,8,4], num_classes)
def darknet_mini(num_classes=1000): return Darknet([1,2,4,4,2], num_classes, start_nf=24)
def darknet_mini2(num_classes=1000): return Darknet([1,2,8,8,4], num_classes, start_nf=16)
def darknet_mini3(num_classes=1000): return Darknet([1,2,4,4], num_classes)

