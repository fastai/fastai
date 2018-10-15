"Utility functions to help deal with tensors"
from .imports.torch import *
from .core import *

AffineMatrix = Tensor
BoolOrTensor = Union[bool,Tensor]
FloatOrTensor = Union[float,Tensor]
IntOrTensor = Union[int,Tensor]
ItemsList = Collection[Union[Tensor,ItemBase,'ItemsList',float,int]]
LambdaFunc = Callable[[Tensor],Tensor]
LayerFunc = Callable[[nn.Module],None]
Model = nn.Module
ModuleList = Collection[nn.Module]
OptOptimizer = Optional[optim.Optimizer]
ParamList = Collection[nn.Parameter]
Rank0Tensor = NewType('OneEltTensor', Tensor)
SplitFunc = Callable[[Model], List[Model]]
SplitFuncOrIdxList = Union[Callable, Collection[ModuleList]]
TensorOrNumber = Union[Tensor,Number]
TensorOrNumList = Collection[TensorOrNumber]
TensorImage = Tensor
TensorImageSize = Tuple[int,int,int]
Tensors = Union[Tensor, Collection['Tensors']]
Weights = Dict[str,Tensor]

AffineFunc = Callable[[KWArgs], AffineMatrix]
HookFunc = Callable[[Model, Tensors, Tensors], Any]
LogitTensorImage = TensorImage
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
MetricFunc = Callable[[Tensor,Tensor],TensorOrNumber]
MetricFuncList = Collection[MetricFunc]
MetricsList = Collection[TensorOrNumber]
OptLossFunc = Optional[LossFunction]
OptMetrics = Optional[MetricsList]
OptSplitFunc = Optional[SplitFunc]
PixelFunc = Callable[[TensorImage, ArgStar, KWArgs], TensorImage]

LightingFunc = Callable[[LogitTensorImage, ArgStar, KWArgs], LogitTensorImage]

fastai_types = {
    AnnealFunc:'AnnealFunc', ArgStar:'ArgStar', BatchSamples:'BatchSamples', Classes:'Classes',
    FilePathList:'FilePathList', Floats:'Floats', ImgLabel:'ImgLabel', ImgLabels:'ImgLabels', KeyFunc:'KeyFunc',
    KWArgs:'KWArgs', ListOrItem:'ListOrItem', ListRules:'ListRules', ListSizes:'ListSizes',
    NPArrayableList:'NPArrayableList', NPArrayList:'NPArrayList', NPArrayMask:'NPArrayMask', NPImage:'NPImage',
    OptDataFrame:'OptDataFrame', OptListOrItem:'OptListOrItem', OptRange:'OptRange', OptStrTuple:'OptStrTuple',
    OptStats:'OptStats', PathOrStr:'PathOrStr', PBar:'PBar', Point:'Point', Points:'Points', Sizes:'Sizes',
    SplitArrayList:'SplitArrayList', StartOptEnd:'StartOptEnd', StrList:'StrList', Tokens:'Tokens',
    OptStrList:'OptStrList', AffineMatrix:'AffineMatrix', BoolOrTensor:'BoolOrTensor', FloatOrTensor:'FloatOrTensor',
    IntOrTensor:'IntOrTensor', ItemsList:'ItemsList', LambdaFunc:'LambdaFunc',
    LayerFunc:'LayerFunc', Model:'Model', ModuleList:'ModuleList', OptOptimizer:'OptOptimizer', ParamList:'ParamList',
    Rank0Tensor:'Rank0Tensor', SplitFunc:'SplitFunc', SplitFuncOrIdxList:'SplitFuncOrIdxList',
    TensorOrNumber:'TensorOrNumber', TensorOrNumList:'TensorOrNumList', TensorImage:'TensorImage',
    TensorImageSize:'TensorImageSize', Tensors:'Tensors', Weights:'Weights', AffineFunc:'AffineFunc',
    HookFunc:'HookFunc', LogitTensorImage:'LogitTensorImage', LossFunction:'LossFunction', MetricFunc:'MetricFunc',
    MetricFuncList:'MetricFuncList', MetricsList:'MetricsList', OptLossFunc:'OptLossFunc', OptMetrics:'OptMetrics',
    OptSplitFunc:'OptSplitFunc', PixelFunc:'PixelFunc', LightingFunc:'LightingFunc',
}

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_default_cpus = min(16, num_cpus())
_default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
defaults = SimpleNamespace(device=_default_device, cpus=_default_cpus)
AdamW = partial(optim.Adam, betas=(0.9,0.99))

def tensor(x:Any)->Tensor:
    "Like `torch.as_tensor`, but handle lists too"
    return torch.tensor(x) if is_listy(x) else as_tensor(x)

def np_address(x:np.ndarray)->int:
    "Address of `x` in memory"
    return x.__array_interface__['data'][0]

def to_detach(b:Tensors):
    "Recursively detach lists of tensors in `b `"
    if is_listy(b): return [to_detach(o) for o in b]
    return b.detach() if isinstance(b,Tensor) else b

def to_data(b:ItemsList):
    "Recursively map lists of items in `b ` to their wrapped data"
    if is_listy(b): return [to_data(o) for o in b]
    return b.data if isinstance(b,ItemBase) else b

def to_device(b:Tensors, device:torch.device):
    "Ensure `b` is on `device`."
    device = ifnone(device, defaults.device)
    if is_listy(b): return [to_device(o, device) for o in b]
    return b.to(device)

def data_collate(batch:ItemsList)->Tensor:
    "Convert `batch` items to tensor data."
    return torch.utils.data.dataloader.default_collate(to_data(batch))

def requires_grad(m:nn.Module, b:Optional[bool]=None)->Optional[bool]:
    "If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param."
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b

def trainable_params(m:nn.Module)->ParamList:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res

def children(m:nn.Module)->ModuleList:
    "Get children of module `m`."
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of children modules in module `m`."
    return len(children(m))

def range_children(m:nn.Module)->Iterator[int]:
    "Return iterator of len of children of `m`."
    return range(num_children(m))

flatten_model=lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]
def first_layer(m:nn.Module)->nn.Module:
    "Retrieve first layer in a module `m`."
    return flatten_model(m)[0]

def split_model_idx(model:nn.Module, idxs:Collection[int])->ModuleList:
    "Split `model` according to the indices in `idxs`."
    layers = flatten_model(model)
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(layers): idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i,j in zip(idxs[:-1],idxs[1:])]

def split_model(model:nn.Module, splits:Collection[Union[Model,ModuleList]], want_idxs:bool=False):
    "Split `model` according to the layers in `splits`."
    layers = flatten_model(model)
    splits = listify(splits)
    if isinstance(splits[0], nn.Module):
        idxs = [layers.index(first_layer(s)) for s in splits]
        res = split_model_idx(model, idxs)
    else: res = [nn.Sequential(*s) for s in splits]
    return (res,idxs) if want_idxs else res

#TODO: add the test to put bias with bn layers
def split_bn_bias(layer_groups:ModuleList)->ModuleList:
    "Sort each layer in  `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        l1,l2 = [],[]
        for c in l.children():
            if isinstance(c, bn_types): l2.append(c)
            else:                       l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups

def set_bn_eval(m:nn.Module)->None:
    "Set bn layers in eval mode for all recursive children of `m`."
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            l.eval()
        set_bn_eval(l)

def to_half(b:Collection[Tensor])->Collection[Tensor]:
    "`b` = [x,y] -> [x.half(),y] (half precision)"
    return [b[0].half(), b[1]]

def bn2float(module:nn.Module)->nn.Module:
    "If `module` is batchnorm don't use half precision."
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:nn.Module)->nn.Module:
    "Convert `model` to half precision except the batchnorm layers."
    return bn2float(model.half())

def cond_init(m:nn.Module, init_func:LayerFunc):
    "Initialize the non-batchnorm layers of `m` with `init_func`"
    if (not isinstance(m, bn_types)) and requires_grad(m):
        if hasattr(m, 'weight'): init_func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)

def apply_leaf(m:nn.Module, f:LayerFunc):
    "Apply `f` to children of `m`."
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)

def apply_init(m, init_func:LayerFunc):
    "Initialize all non-batchnorm layers of `m` with `init_func`."
    apply_leaf(m, partial(cond_init, init_func=init_func))

def in_channels(m:Model) -> List[int]:
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')

def calc_loss(y_pred:Tensor, y_true:Tensor, loss_class:type=nn.CrossEntropyLoss, bs=64):
    "Calculate loss between `y_pred` and `y_true` using `loss_class` and `bs`."
    loss_dl = DataLoader(TensorDataset(as_tensor(y_pred),as_tensor(y_true)), bs)
    with torch.no_grad():
        return torch.cat([loss_class(reduction='none')(*b) for b in loss_dl])

def to_np(x): return x.cpu().numpy()

def model_type(dtype):
    return (torch.float32 if np.issubdtype(dtype, np.floating) else
            torch.int64 if np.issubdtype(dtype, np.integer)
            else None)

def np2model_tensor(a):
    dtype = model_type(a.dtype)
    res = as_tensor(a)
    if not dtype: return res
    return res.type(dtype)

def trange_of(x): return torch.arange(len(x))
