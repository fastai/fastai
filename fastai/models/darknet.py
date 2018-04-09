import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(nn.Module):
    "convolutional layer then batchnorm"

    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)

class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x): return self.conv2(self.conv1(x)) + x

class Darknet(nn.Module):
    "Replicates the darknet classifier from the YOLOv3 paper (table 1)"

    def __init__(self, num_blocks, num_classes=1000, start_nf=32):
        super().__init__()
        self.conv = ConvBN(3, start_nf, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_group_layer(start_nf, num_blocks[0])
        self.layer2 = self.make_group_layer(start_nf*2, num_blocks[1], stride=2)
        self.layer3 = self.make_group_layer(start_nf*4, num_blocks[2], stride=2)
        self.layer4 = self.make_group_layer(start_nf*8, num_blocks[3], stride=2)
        self.layer5 = self.make_group_layer(start_nf*16,num_blocks[4], stride=2)
        self.linear = nn.Linear(start_nf*32, num_classes)

    def make_group_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in,ch_in*2,stride=stride)]
        for i in range(num_blocks): layers.append(DarknetBlock(ch_in*2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def darknet_53(num_classes=1000):    return Darknet([1,2,8,8,4], num_classes)
def darknet_small(num_classes=1000): return Darknet([1,2,4,4,2], num_classes)
def darknet_mini(num_classes=1000): return Darknet([1,2,8,8,4], num_classes, start_nf=24)

