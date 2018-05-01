import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from ..layers import *

__all__ = ['vgg_resnet50']

model_urls = {
    'vgg_resnet50': 'https://download.pytorch.org/models/vggresnet.pth',
}


def conv(ni, nf, ks=3, stride=1):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn1(planes):
    m = nn.BatchNorm1d(planes)
    m.weight.data.fill_(1)
    m.bias.data.zero_()
    return m

def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, stride=stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)

        out = residual + out
        out = self.relu(out)
        out = self.bn2(out)

        return out


class BottleneckFinal(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, ks=1)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, stride=stride)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, planes*4, ks=1)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = residual + out
        out = self.bn3(out)
        out = self.relu(out)

        return out

class BottleneckZero(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, ks=1)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, stride=stride)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, planes*4, ks=1)
        self.bn3 = bn(planes * 4, init_zero=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = residual + out
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv(inplanes, planes, ks=1)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, stride=stride)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, planes*4, ks=1)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = residual + out
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, k=1, vgg_head=False):
        super().__init__()
        self.inplanes = 64

        features = [conv(3, 64, ks=7, stride=2)
            , bn(64) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            , self._make_layer(block, int(64*k), layers[0])
            , self._make_layer(block, int(128*k), layers[1], stride=2)
            , self._make_layer(block, int(256*k), layers[2], stride=2)
            , self._make_layer(block, int(512*k), layers[3], stride=2)]
        out_sz = int(512*k) * block.expansion

        if vgg_head:
            features += [nn.AdaptiveAvgPool2d(3), Flatten()
                , nn.Linear(out_sz*3*3, 4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096,   4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096, num_classes)]
        else: features += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(out_sz, num_classes)]

        self.features = nn.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes*block.expansion, ks=1, stride=stride),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x): return self.features(x)


def bnf_resnet50 (): return ResNet(BottleneckFinal, [3, 4, 6, 3])
def bnz_resnet50 (): return ResNet(BottleneckZero, [3, 4, 6, 3])
def w5_resnet50 (): return ResNet(Bottleneck, [2, 3, 3, 2], k=1.5)
def w25_resnet50(): return ResNet(Bottleneck, [3, 4, 4, 3], k=1.25)
def w125_resnet50(): return ResNet(Bottleneck, [3, 4, 6, 3], k=1.125)
def vgg_resnet34(): return ResNet(BasicBlock, [3, 4, 6, 3], vgg_head=True)
def vgg_resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], vgg_head=True)
    if pretrained: model.load_state_dict(torch.load('/home/jhoward/.torch/models/vgg_resnet50.pth'))
    return model

