from .imports import *
from .torch_imports import *

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

def _make_bipolar(fn):
    """ based on code from https://github.com/pytorch/pytorch/issues/4281 
        by larspars (https://github.com/larspars)
    
    Usage: 
    @_make_bipolar
    def bReLU(x):
        return nn.ReLU()(x)
    
    input = torch.autograd.Variable(torch.randn(3))
    print(input)
    print(nn.ReLU()(input))
    print(bReLU(input))

    Variable containing:
    -0.4373
    1.5798
    -0.2647
    -1.0417
    0.7401
    [torch.FloatTensor of size 5]

    Variable containing:
    0.0000
    1.5798
    0.0000
    0.0000
    0.7401
    [torch.FloatTensor of size 5]

    Variable containing:
    0.0000
    1.5798
    0.0000
    -1.0417
    0.0000
    [torch.FloatTensor of size 5]

    """
    def _fn(x, *args, **kwargs):
        dim = 0 if x.dim() == 1 else 1
        x0, x1 = torch.chunk(x, chunks=2, dim=dim)
        y0 = fn(x0, *args, **kwargs)
        y1 = torch.matmul(V(torch.FloatTensor(np.identity(int(x.shape[0] / 2.0))*-1.0)), fn(-x1, *args, **kwargs))
        return torch.cat((y0, y1), dim=dim)

    return _fn




