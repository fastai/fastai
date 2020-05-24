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
    """Take out conv part of a given efficientnet model"""
    def __init__(self, model: EfficientNet, cut: Optional[int]):
        super().__init__()
        self.model = deepcopy(model)
        self.model._blocks = self.model._blocks[:cut]

    def forward(self, inputs):
        return self.model.extract_features(inputs)

