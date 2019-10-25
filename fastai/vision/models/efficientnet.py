from ...core import *

pretrainedmodels = try_import('efficientnet_pytorch') 

if not pretrainedmodels:
    raise Exception('Error: efficientnet-pytorch is needed. pip install efficientnet-pytorch')
from efficientnet_pytorch import EfficientNet

def EfficientNetB1(data): return EfficientNet.from_pretrained('efficientnet-b1', num_classes=data.c)
def EfficientNetB2(data): return EfficientNet.from_pretrained('efficientnet-b2', num_classes=data.c)
def EfficientNetB3(data): return EfficientNet.from_pretrained('efficientnet-b3', num_classes=data.c)
def EfficientNetB4(data): return EfficientNet.from_pretrained('efficientnet-b4', num_classes=data.c)
def EfficientNetB5(data): return EfficientNet.from_pretrained('efficientnet-b5', num_classes=data.c)
def EfficientNetB6(data): return EfficientNet.from_pretrained('efficientnet-b6', num_classes=data.c)
def EfficientNetB7(data): return EfficientNet.from_pretrained('efficientnet-b7', num_classes=data.c)
