import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from functools import partial

__all__ = ['XResNet', 'xresnet18', 'xresnet34', 'xresnet50', 'xresnet101', 'xresnet152']

# TODO: ELU init (a=0.54; gain=1.55)
act_fn = nn.ReLU(inplace=True)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    # TODO init final linear bias to return 1/c (with log etc)
    # TODO linear weight should be kaiming or something?
    for l in m.children(): init_cnn(l)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1):
        super().__init__()
        nf,ni = nh*expansion,ni*expansion
        self.convs = nn.Sequential(
            noop if expansion==1 else conv_layer(ni, nh, 1),
            conv_layer(ni if expansion==1 else nh, nh, stride=stride),
            conv_layer(nh, nf, 3 if expansion==1 else 1, zero_bn=True, act=False))
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1)
        self.pool = noop if stride==1 else nn.AvgPool2d(2)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))

class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, num_classes=1000):
        block_szs = [64//expansion,64,128,256,512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2)
                  for i,l in enumerate(layers)]
        super().__init__(
            conv_layer(3, 16, stride=2), conv_layer(16, 32), conv_layer(32, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(block_szs[-1]*expansion, num_classes),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(blocks)])

def xresnet(expansion, n_layers, name, pre=False, **kwargs):
    model = XResNet(expansion, n_layers, **kwargs)
    #if pre: model.load_state_dict(model_zoo.load_url(model_urls[name]))
    if pre: model.load_state_dict(torch.load(model_urls[name]))
    return model

def xresnet18(pretrained=False, **kwargs):
    return xresnet(1, [2, 2, 2, 2], 'xresnet18', pre=pretrained, **kwargs)

def xresnet34(pretrained=False, **kwargs):
    return xresnet(1, [3, 4, 6, 3], 'xresnet34', pre=pretrained, **kwargs)

def xresnet50(pretrained=False, **kwargs):
    return xresnet(4, [3, 4, 6, 3], 'xresnet50', pre=pretrained, **kwargs)

def xresnet101(pretrained=False, **kwargs):
    return xresnet(4, [3, 4, 23, 3], 'xresnet101', pre=pretrained, **kwargs)

def xresnet152(pretrained=False, **kwargs):
    return xresnet(4, [3, 8, 36, 3], 'xresnet152', pre=pretrained, **kwargs)

