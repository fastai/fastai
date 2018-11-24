"`fastai.layers` provides essential functions to building and modifying `model` architectures"
from .torch_core import *

__all__ = ['AdaptiveConcatPool2d', 'MSELossFlat', 'CrossEntropyFlat', 'Debugger', 'Flatten', 'Lambda', 'PoolFlatten', 'ResizeBatch',
           'bn_drop_lin', 'conv2d', 'conv2d_trans', 'conv_layer', 'embedding', 'simple_cnn',
           'std_upsample_head', 'trunc_normal_', 'PixelShuffle_ICNR', 'icnr', 'NoopLoss']

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func:LambdaFunc):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def ResizeBatch(*size:int) -> Tensor:
    "Layer that resizes x to `size`, good for connecting mismatched layers."
    return Lambda(lambda x: x.view((-1,)+size))

def Flatten()->Tensor:
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))

def PoolFlatten()->nn.Sequential:
    "Apply `nn.AdaptiveAvgPool2d` to `x` and then flatten the result."
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def conv2d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False) -> nn.Conv2d:
    "Create `nn.Conv2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to `k//2`."
    if padding is None: padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def conv2d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose2d:
    "Create `nn.ConvTranspose2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size, `stride`: stride. `padding` defaults to 0."
    return nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, bn:bool=True,
               leaky:float=None, transpose:bool=False, wn:bool=False, init:Callable=nn.init.kaiming_normal_):
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv2d
    activ = nn.LeakyReLU(inplace=True, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=True)
    conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding)
    if init:
        init(conv.weight)
        if hasattr(conv, 'bias') and hasattr(conv.bias, 'data'): conv.bias.data.fill_(0.)
    if wn:
        bn=False
        conv = weight_norm(conv)
    layers = [conv, activ]
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Debugger(nn.Module):
    "A module to debug inside a model."
    def forward(self,x:Tensor) -> Tensor:
        set_trace()
        return x

def std_upsample_head(c, *nfs:Collection[int]) -> nn.Module:
    "Create a sequence of upsample layers."
    return nn.Sequential(
        nn.ReLU(),
        *(conv_layer(nfs[i],nfs[i+1],ks=2, stride=2, padding=0, transpose=True) for i in range(4)),
        conv2d_trans(nfs[-1], c)
    )

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2):
        nf = ifnone(nf, ni)
        conv = weight_norm(conv2d(ni, nf * (scale**2), ks=1))
        icnr(conv.weight)
        shuf = nn.PixelShuffle(scale)
        return super().__init__(conv,shuf)

class CrossEntropyFlat(nn.CrossEntropyLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        n,c,*_ = input.shape
        return super().forward(input.view(n, c, -1), target.view(n, -1))

class MSELossFlat(nn.MSELoss):
    "Same as `nn.MSELoss`, but flattens input and target."
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        return super().forward(input.view(-1), target.view(-1))

class NoopLoss(nn.Module):
    "Just returns the `output`."
    def forward(self, output, target): return output[0]

def simple_cnn(actns:Collection[int], kernel_szs:Collection[int]=None,
               strides:Collection[int]=None, bn=False) -> nn.Sequential:
    "CNN with `conv_layer` defined by `actns`, `kernel_szs` and `strides`, plus batchnorm if `bn`."
    nl = len(actns)-1
    kernel_szs = ifnone(kernel_szs, [3]*nl)
    strides    = ifnone(strides   , [2]*nl)
    layers = [conv_layer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i],
              bn=(bn and i<(len(strides)-1))) for i in range_of(strides)]
    layers.append(PoolFlatten())
    return nn.Sequential(*layers)

def trunc_normal_(x:Tensor, mean:float=0., std:float=1.) -> Tensor:
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def embedding(ni:int,nf:int) -> nn.Module:
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb
