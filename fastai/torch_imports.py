import os
from distutils.version import LooseVersion
import torch, torchvision, torchtext
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torch.nn.init import kaiming_uniform, kaiming_normal
from torchvision.transforms import Compose
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg16_bn, vgg19_bn
from torchvision.models import densenet121, densenet161, densenet169, densenet201

from .models.resnext_50_32x4d import resnext_50_32x4d
from .models.resnext_101_32x4d import resnext_101_32x4d
from .models.resnext_101_64x4d import resnext_101_64x4d
from .models.wrn_50_2f import wrn_50_2f
from .models.inceptionresnetv2 import InceptionResnetV2
from .models.inceptionv4 import inceptionv4
from .models.nasnet import nasnetalarge
from .models.fa_resnet import *

import warnings
warnings.filterwarnings('ignore', message='Implicit dimension choice', category=UserWarning)

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
if IS_TORCH_04:
    from torch.nn.init import kaiming_uniform_ as kaiming_uniform
    from torch.nn.init import kaiming_normal_ as kaiming_normal

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m

def _fastai_model(name, paper_title, paper_href):
    def add_docs_wrapper(f):
        f.__doc__ = f"""{name} model from
        `"{paper_title}" <{paper_href}>`_

        Args:
           pre (bool): If True, returns a model pre-trained on ImageNet
        """
        return f
    return add_docs_wrapper

@_fastai_model('Inception 4', 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning',
               'https://arxiv.org/pdf/1602.07261.pdf')
def inception_4(pre): return children(inceptionv4(pretrained=pre))[0]

@_fastai_model('Inception 4', 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning',
               'https://arxiv.org/pdf/1602.07261.pdf')
def inceptionresnet_2(pre): return load_pre(pre, InceptionResnetV2, 'inceptionresnetv2-520b38e4')

@_fastai_model('ResNeXt 50', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext50(pre): return load_pre(pre, resnext_50_32x4d, 'resnext_50_32x4d')

@_fastai_model('ResNeXt 101_32', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext101(pre): return load_pre(pre, resnext_101_32x4d, 'resnext_101_32x4d')

@_fastai_model('ResNeXt 101_64', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext101_64(pre): return load_pre(pre, resnext_101_64x4d, 'resnext_101_64x4d')

@_fastai_model('Wide Residual Networks', 'Wide Residual Networks',
               'https://arxiv.org/pdf/1605.07146.pdf')
def wrn(pre): return load_pre(pre, wrn_50_2f, 'wrn_50_2f')

@_fastai_model('Densenet-121', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn121(pre): return children(densenet121(pre))[0]

@_fastai_model('Densenet-169', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn161(pre): return children(densenet161(pre))[0]

@_fastai_model('Densenet-161', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn169(pre): return children(densenet169(pre))[0]

@_fastai_model('Densenet-201', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn201(pre): return children(densenet201(pre))[0]

@_fastai_model('Vgg-16 with batch norm added', 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
               'https://arxiv.org/pdf/1409.1556.pdf')
def vgg16(pre): return children(vgg16_bn(pre))[0]

@_fastai_model('Vgg-19 with batch norm added', 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
               'https://arxiv.org/pdf/1409.1556.pdf')
def vgg19(pre): return children(vgg19_bn(pre))[0]

