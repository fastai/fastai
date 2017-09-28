from .imports import *
from .torch_imports import *
from .core import *

def opt_params(parm, lr, wd):
    return {'params': chain_params(parm), 'lr':lr, 'weight_decay':wd}

class LayerOptimizer():
    def __init__(self, opt_fn, layer_groups, lrs, wds=None):
        if not isinstance(layer_groups, (list,tuple)): layer_groups=[layer_groups]
        if not isinstance(lrs, Iterable): lrs=[lrs]
        if len(lrs)==1: lrs=lrs*len(layer_groups)
        if wds is None: wds=0.
        if not isinstance(wds, Iterable): wds=[wds]
        if len(wds)==1: wds=wds*len(layer_groups)
        self.layer_groups,self.lrs,self.wds = layer_groups,lrs,wds
        self.opt = opt_fn(self.opt_params())

    def opt_params(self):
        params = list(zip(self.layer_groups,self.lrs,self.wds))
        return [opt_params(*p) for p in params]

    @property
    def lr(self): return self.lrs[-1]

    def set_lrs(self, lrs):
        self.lrs=lrs
        set_lrs(self.opt, lrs)

def set_lrs(opt, lrs):
    if not isinstance(lrs, Iterable): lrs=[lrs]
    if len(lrs)==1: lrs=lrs*len(opt.param_groups)
    for pg,lr in zip(opt.param_groups,lrs): pg['lr'] = lr
