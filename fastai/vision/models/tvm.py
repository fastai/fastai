from torchvision.models import *
import types as _t

_g = globals()
for _k, _v in list(_g.items()):
    if (
        isinstance(_v, _t.ModuleType) and _v.__name__.startswith("torchvision.models")
    ) or (callable(_v) and _v.__module__ == "torchvision.models._api"):
        del _g[_k]

del _k, _v, _g, _t
