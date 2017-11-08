import os
import torch, torchvision, torchtext
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torch.nn.init import kaiming_uniform, kaiming_normal
from torchvision.transforms import Compose
from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg16
from torchvision.models import densenet121, densenet161, densenet169, densenet201

from .models.resnext_50_32x4d import resnext_50_32x4d
from .models.resnext_101_32x4d import resnext_101_32x4d
from .models.resnext_101_64x4d import resnext_101_64x4d
from .models.wrn_50_2f import wrn_50_2f
from .models.inceptionresnetv2 import InceptionResnetV2
from .models.inceptionv4 import InceptionV4

def children(m): return list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m

def inception_4(pre):
    return children(load_pre(pre, InceptionV4, 'inceptionv4-97ef9c30'))[0]
def inceptionresnet_2(pre): return load_pre(pre, InceptionResnetV2, 'inceptionresnetv2-d579a627')
def resnext50(pre): return load_pre(pre, resnext_50_32x4d, 'resnext_50_32x4d')
def resnext101(pre): return load_pre(pre, resnext_101_32x4d, 'resnext_101_32x4d')
def resnext101_64(pre): return load_pre(pre, resnext_101_64x4d, 'resnext_101_64x4d')
def wrn(pre): return load_pre(pre, wrn_50_2f, 'wrn_50_2f')
def dn121(pre): return children(densenet121(pre))[0]
def dn161(pre): return children(densenet161(pre))[0]
def dn169(pre): return children(densenet169(pre))[0]
def dn201(pre): return children(densenet201(pre))[0]

def vgg_surrogate(pretrained=True):
    '''
    Method returns a surrogate version of the VGG
    where all the layers are initialized as a simple
    sequence of constituent layers. The pytorch out-
    of-the-box vgg16 model comes as nested Sequential
    models of the convolution layers and fully-connected
    layers respectively. Working with the below makes
    it much easier to manipulate the independent layers,
    as we do it in the ConvnetBuilder class.
    :param pretrained:
    :return:
    '''
    vgg = vgg16(pretrained=pretrained)

    surrogateList = []

    for layer in list(vgg.children())[0]:
        surrogateList.append(layer)

    for layer in list(vgg.children())[1]:
        surrogateList.append(layer)

    return nn.Sequential(*surrogateList)

