try:
    import pretrainedmodels
    _ = pretrainedmodels.inceptionv4
except (ImportError, AttributeError) as e:
    print('Error: you must install library \"pretrainedmodels\" to use the model. Try:\n' + 
          '> pip install pretrainedmodels \n')
    raise

from torch import nn
from ..learner import model_meta

__all__ = ['inceptionv4', 'inceptionresnetv2', 'nasnetamobile', 'dpn92', 'xception_cadene', 'se_resnet50', 
           'se_resnet101', 'se_resnext50_32x4d', 'senet154', 'pnasnet5large']

def inceptionv4(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionv4(pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])

model_meta[inceptionv4] = {'cut': -2, 'split': lambda m: (m[0][11], m[1])}

def inceptionresnetv2(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
    return nn.Sequential(*model.children())

model_meta[inceptionresnetv2] = {'cut': -2, 'split': lambda m: (m[0][9], m[1])}

def identity(x): return x

def nasnetamobile(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.nasnetamobile(pretrained=pretrained, num_classes=1000)
    model.logits = identity
    return nn.Sequential(model)

model_meta[nasnetamobile] = {'cut': None, 'split': lambda m: (list(m[0][0].children())[8], m[1])}

def dpn92(pretrained=False):
    pretrained = 'imagenet+5k' if pretrained else None
    model = pretrainedmodels.dpn92(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))

model_meta[dpn92] = {'cut': -1, 'split': lambda m: (m[0][0][16], m[1])}

def xception_cadene(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.xception(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))

model_meta[xception_cadene] = {'cut': -1, 'split': lambda m: (m[0][11], m[1])}

def se_resnet50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnet50(pretrained=pretrained)
    return model

def se_resnet101(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnet101(pretrained=pretrained)
    return model

def se_resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return model

def senet154(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.senet154(pretrained=pretrained)
    return model

_se_resnet_meta = {'cut': -2, 'split': lambda m: (m[0][3], m[1])}
model_meta[se_resnet50] = _se_resnet_meta
model_meta[se_resnet101] = _se_resnet_meta
model_meta[se_resnext50_32x4d] = _se_resnet_meta
model_meta[senet154] = {'cut': -3, 'split': lambda m: (m[0][3], m[1])}

def pnasnet5large(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.pnasnet5large(pretrained=pretrained, num_classes=1000)
    model.logits = identity
    return nn.Sequential(model)

model_meta[pnasnet5large] = {'cut': None, 'split': lambda m: (list(m[0][0].children())[8], m[1])}

# TODO: add "resnext101_32x4d" "resnext101_64x4d" after serialization issue is fixed:
# https://github.com/Cadene/pretrained-models.pytorch/pull/128
