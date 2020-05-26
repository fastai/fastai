from torch import nn

from ...core import *

pretrainedmodels = try_import('efficientnet_pytorch') 

if not pretrainedmodels:
    raise Exception('Error: efficientnet-pytorch is needed. pip install efficientnet-pytorch')
from efficientnet_pytorch import EfficientNet

def create_efficientnet(model_name, pretrained=False):
    model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
    return EfficientNetWrapper(model)

def EfficientNetB0(pretrained=False): return create_efficientnet("efficientnet-b0", pretrained)
def EfficientNetB1(pretrained=False): return create_efficientnet("efficientnet-b1", pretrained)
def EfficientNetB2(pretrained=False): return create_efficientnet("efficientnet-b2", pretrained)
def EfficientNetB3(pretrained=False): return create_efficientnet("efficientnet-b3", pretrained)
def EfficientNetB4(pretrained=False): return create_efficientnet("efficientnet-b4", pretrained)
def EfficientNetB5(pretrained=False): return create_efficientnet("efficientnet-b5", pretrained)
def EfficientNetB6(pretrained=False): return create_efficientnet("efficientnet-b6", pretrained)
def EfficientNetB7(pretrained=False): return create_efficientnet("efficientnet-b7", pretrained)

class EfficientnetBlocks(nn.Module):

    def __init__(self, model: EfficientNet):
        super().__init__()
        self._blocks = deepcopy(model._blocks)
        self._global_params = deepcopy(model._global_params)

    def forward(self, inputs):
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            inputs = block(inputs, drop_connect_rate=drop_connect_rate)
        return inputs

    def __getitem__(self, item):
        return self._blocks[item]

    def __len__(self):
        return len(self._blocks)

class EfficientNetWrapper(nn.Module):

    def __init__(self, model: EfficientNet):
        super().__init__()
        self.model = deepcopy(model)
        self.in_features = model._fc.in_features
        self.model._blocks = EfficientnetBlocks(model)

    def children(self):
        swish = self.model._swish
        modules = list(self.model.children())
        modules.remove(swish)
        modules.insert(modules.index(self.model._bn0) + 1, swish)
        modules.insert(modules.index(self.model._bn1) + 1, swish)
        return (module for module in deepcopy(modules))

    def __getitem__(self, item):
        return nn.Sequential(*self.children())[item]

    def __len__(self):
        return len(list(self.children()))
