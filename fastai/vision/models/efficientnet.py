from torch import nn

from ...core import *

pretrainedmodels = try_import('efficientnet_pytorch') 

if not pretrainedmodels:
    raise Exception('Error: efficientnet-pytorch is needed. pip install efficientnet-pytorch')
from efficientnet_pytorch import EfficientNet

def EfficientNetB0(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
def EfficientNetB1(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b1') if pretrained else EfficientNet.from_name('efficientnet-b1')
def EfficientNetB2(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b2') if pretrained else EfficientNet.from_name('efficientnet-b2')
def EfficientNetB3(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b3') if pretrained else EfficientNet.from_name('efficientnet-b3')
def EfficientNetB4(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b4') if pretrained else EfficientNet.from_name('efficientnet-b4')
def EfficientNetB5(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b5') if pretrained else EfficientNet.from_name('efficientnet-b5')
def EfficientNetB6(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b6') if pretrained else EfficientNet.from_name('efficientnet-b6')
def EfficientNetB7(pretrained=False): return EfficientNet.from_pretrained('efficientnet-b7') if pretrained else EfficientNet.from_name('efficientnet-b7')


class EfficientNetBody(nn.Module):

    def __init__(self, model: EfficientNet):
        super().__init__()
        self._swish = deepcopy(model._swish)
        self._bn0 = deepcopy(model._bn0)
        self._conv_stem = deepcopy(model._conv_stem)
        self._blocks = deepcopy(model._blocks)
        self._drop_connect_rate = deepcopy(model._global_params.drop_connect_rate)
        self._bn1 = deepcopy(model._bn1)
        self._conv_head = deepcopy(model._conv_head)

    def forward(self, inputs):
        """ The extract_features method of EfficientNet.

         Returns output of the final convolution layer.
         """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

