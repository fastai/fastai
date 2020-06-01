from fastai.layers import Flatten
from torch import nn

from ...core import *

pretrainedmodels = try_import('efficientnet_pytorch') 

if not pretrainedmodels:
    raise Exception('Error: efficientnet-pytorch is needed. pip install efficientnet-pytorch')
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

def create_efficientnet(model_name, pretrained=False):
    model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
    return SequentialEfficientNet(model)

def EfficientNetB0(pretrained=False): return create_efficientnet("efficientnet-b0", pretrained)
def EfficientNetB1(pretrained=False): return create_efficientnet("efficientnet-b1", pretrained)
def EfficientNetB2(pretrained=False): return create_efficientnet("efficientnet-b2", pretrained)
def EfficientNetB3(pretrained=False): return create_efficientnet("efficientnet-b3", pretrained)
def EfficientNetB4(pretrained=False): return create_efficientnet("efficientnet-b4", pretrained)
def EfficientNetB5(pretrained=False): return create_efficientnet("efficientnet-b5", pretrained)
def EfficientNetB6(pretrained=False): return create_efficientnet("efficientnet-b6", pretrained)
def EfficientNetB7(pretrained=False): return create_efficientnet("efficientnet-b7", pretrained)

class MBConvBlockWrapper(nn.Module):

    def __init__(self, model: MBConvBlock, drop_connect_rate: float=None):
        super().__init__()
        self.model = model
        self.drop_connect_rate = drop_connect_rate

    def forward(self, inputs):
        return self.model.forward(inputs, self.drop_connect_rate)

class SequentialEfficientNet(nn.Module):
    """Convert EfficientNet instance to an object that can be processed by existing cnn_learner scheme."""

    def __init__(self, model: EfficientNet):
        super().__init__()
        _transform_conv_2d_static_same_padding_to_sequential(model)
        for block in model._blocks:
            _transform_conv_2d_static_same_padding_to_sequential(block)
        self._conv_stem = model._conv_stem
        self._bn0 = model._bn0
        self._swish0 = deepcopy(model._swish)
        blocks = []
        for idx, block in enumerate(model._blocks):
            _transform_conv_2d_static_same_padding_to_sequential(block)
            drop_connect_rate = model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(model._blocks)
            blocks.append(MBConvBlockWrapper(block, drop_connect_rate))
        self._blocks = nn.Sequential(*blocks)
        self._conv_head = model._conv_head
        self._bn1 = model._bn1
        self._swish1 = deepcopy(model._swish)
        self._avg_pooling = model._avg_pooling
        self._flatten = Flatten()
        self._dropout = model._dropout
        self._fc = model._fc

    def forward(self, inputs):
        return nn.Sequential(*self.children()).forward(inputs)

def _transform_conv_2d_static_same_padding_to_sequential(model: nn.Module):
    """Split Conv2dStaticSamePadding instance into Sequential(padding, Conv2d)."""
    for name, module in model._modules.items():
        if isinstance(module, Conv2dStaticSamePadding):
            conv2d_arguments = [param.name for param in inspect.signature(nn.Conv2d).parameters.values()]
            filtered_dict = dict(filter(lambda attr: attr[0] in conv2d_arguments, module.__dict__.items()))
            conv2d = nn.Conv2d(bias=module.bias is not None, **filtered_dict)
            conv2d.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, nn.Sequential(module.static_padding, conv2d))

# From: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'EfficientNetB0': (1.0, 1.0, 224, 0.2),
        'EfficientNetB1': (1.0, 1.1, 240, 0.2),
        'EfficientNetB2': (1.1, 1.2, 260, 0.3),
        'EfficientNetB3': (1.2, 1.4, 300, 0.3),
        'EfficientNetB4': (1.4, 1.8, 380, 0.4),
        'EfficientNetB5': (1.6, 2.2, 456, 0.4),
        'EfficientNetB6': (1.8, 2.6, 528, 0.5),
        'EfficientNetB7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]
