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
ModuleList = Collection[nn.Module]
NPArray = np.ndarray
OptOptimizer = Optional[optim.Optimizer]
ParamList = Collection[nn.Parameter]
Rank0Tensor = NewType('OneEltTensor', Tensor)
SplitFunc = Callable[[nn.Module], List[nn.Module]]
SplitFuncOrIdxList = Union[Callable, Collection[ModuleList]]
TensorOrNumber = Union[Tensor,Number]
TensorOrNumList = Collection[TensorOrNumber]
TensorImage = Tensor
TensorImageSize = Tuple[int,int,int]
Tensors = Union[Tensor, Collection['Tensors']]
Weights = Dict[str,Tensor]

AffineFunc = Callable[[KWArgs], AffineMatrix]
HookFunc = Callable[[nn.Module, Tensors, Tensors], Any]
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
    AnnealFunc:'AnnealFunc', ArgStar:'ArgStar', BatchSamples:'BatchSamples',
    FilePathList:'FilePathList', Floats:'Floats', ImgLabel:'ImgLabel', ImgLabels:'ImgLabels', KeyFunc:'KeyFunc',
    KWArgs:'KWArgs', ListOrItem:'ListOrItem', ListRules:'ListRules', ListSizes:'ListSizes',
    NPArrayableList:'NPArrayableList', NPArrayList:'NPArrayList', NPArrayMask:'NPArrayMask', NPImage:'NPImage',
    OptDataFrame:'OptDataFrame', OptListOrItem:'OptListOrItem', OptRange:'OptRange', OptStrTuple:'OptStrTuple',
    OptStats:'OptStats', PathOrStr:'PathOrStr', PBar:'PBar', Point:'Point', Points:'Points', Sizes:'Sizes',
    SplitArrayList:'SplitArrayList', StartOptEnd:'StartOptEnd', StrList:'StrList', Tokens:'Tokens',
    OptStrList:'OptStrList', AffineMatrix:'AffineMatrix', BoolOrTensor:'BoolOrTensor', FloatOrTensor:'FloatOrTensor',
    IntOrTensor:'IntOrTensor', ItemsList:'ItemsList', LambdaFunc:'LambdaFunc',
    LayerFunc:'LayerFunc', ModuleList:'ModuleList', OptOptimizer:'OptOptimizer', ParamList:'ParamList',
    Rank0Tensor:'Rank0Tensor', SplitFunc:'SplitFunc', SplitFuncOrIdxList:'SplitFuncOrIdxList',
    TensorOrNumber:'TensorOrNumber', TensorOrNumList:'TensorOrNumList', TensorImage:'TensorImage',
    TensorImageSize:'TensorImageSize', Tensors:'Tensors', Weights:'Weights', AffineFunc:'AffineFunc',
    HookFunc:'HookFunc', LogitTensorImage:'LogitTensorImage', LossFunction:'LossFunction', MetricFunc:'MetricFunc',
    MetricFuncList:'MetricFuncList', MetricsList:'MetricsList', OptLossFunc:'OptLossFunc', OptMetrics:'OptMetrics',
    OptSplitFunc:'OptSplitFunc', PixelFunc:'PixelFunc', LightingFunc:'LightingFunc', IntsOrStrs:'IntsOrStrs'
}

torch.set_num_threads(4) # OpenMP doesn't generally like too many threads

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
bias_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
AdamW = partial(optim.Adam, betas=(0.9,0.99))

def tensor(x:Any, *rest)->Tensor:
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,)+rest
    # XXX: Pytorch bug in dataloader using num_workers>0; TODO: create repro and report
    if is_listy(x) and len(x)==0: return tensor(0)
    res = torch.tensor(x) if is_listy(x) else as_tensor(x)
    if res.dtype is torch.int32:
        warn('Tensor is int32: upgrading to int64; for better performance use int64 input')
        return res.long()
    return res

def np_address(x:np.ndarray)->int:
    "Address of `x` in memory."
    return x.__array_interface__['data'][0]

def to_detach(b:Tensors, cpu:bool=True):
    "Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`."
    if is_listy(b): return [to_detach(o, cpu) for o in b]
    if not isinstance(b,Tensor): return b
    b = b.detach()
    return b.cpu() if cpu else b

def to_data(b:ItemsList):
    "Recursively map lists of items in `b ` to their wrapped data."
    if is_listy(b): return [to_data(o) for o in b]
    return b.data if isinstance(b,ItemBase) else b

def to_cpu(b:ItemsList):
    "Recursively map lists of tensors in `b ` to the cpu."
    if is_listy(b): return [to_cpu(o) for o in b]
    return b.cpu() if isinstance(b,Tensor) else b

def to_half(b:Collection[Tensor])->Collection[Tensor]:
    "Recursively map lists of tensors in `b ` to FP16."
    if is_listy(b): return [to_half(o) for o in b]
    return b.half() if b.dtype not in [torch.int64, torch.int32, torch.int16] else b

def to_float(b:Collection[Tensor])->Collection[Tensor]:
    "Recursively map lists of tensors in `b ` to FP16."
    if is_listy(b): return [to_float(o) for o in b]
    return b.float() if b.dtype not in [torch.int64, torch.int32, torch.int16] else b

def to_device(b:Tensors, device:torch.device):
    "Recursively put `b` on `device`."
    device = ifnone(device, defaults.device)
    if is_listy(b): return [to_device(o, device) for o in b]
    if is_dict(b): return {k: to_device(v, device) for k, v in b.items()}
    return b.to(device)

def data_collate(batch:ItemsList)->Tensor:
    "Convert `batch` items to tensor data."
    return torch.utils.data.dataloader.default_collate(to_data(batch))

def requires_grad(m:nn.Module, b:Optional[bool]=None)->Optional[bool]:
    "If `b` is not set return `requires_grad` of first param, else set `requires_grad` on all params as `b`"
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b

def trainable_params(m:nn.Module)->ParamList:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res

def children(m:nn.Module)->ModuleList:
    "Get children of `m`."
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of children modules in `m`."
    return len(children(m))

def range_children(m:nn.Module)->Iterator[int]:
    "Return iterator of len of children of `m`."
    return range(num_children(m))

class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p:nn.Parameter):
        super().__init__()
        self.val = p
    
    def forward(self, x): return x
    
def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children

flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if num_children(m) else [m]

def first_layer(m:nn.Module)->nn.Module:
    "Retrieve first layer in a module `m`."
    return flatten_model(m)[0]

def last_layer(m:nn.Module)->nn.Module:
    "Retrieve last layer in a module `m`."
    return flatten_model(m)[-1]

def split_model_idx(model:nn.Module, idxs:Collection[int])->ModuleList:
    "Split `model` according to the indexes in `idxs`."
    layers = flatten_model(model)
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(layers): idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i,j in zip(idxs[:-1],idxs[1:])]

def split_model(model:nn.Module, splits:Collection[Union[nn.Module,ModuleList]], want_idxs:bool=False):
    "Split `model` according to the layers in `splits`."
    layers = flatten_model(model)
    splits = listify(splits)
    if isinstance(splits[0], nn.Module):
        idxs = [layers.index(first_layer(s)) for s in splits]
        res = split_model_idx(model, idxs)
    else: res = [nn.Sequential(*s) for s in splits]
    return (res,idxs) if want_idxs else res

def split_no_wd_params(layer_groups:Collection[nn.Module])->List[List[nn.Parameter]]:
    "Separate the parameters in `layer_groups` between batchnorm (`bn_types`) and  bias (`bias_types`) from the rest."
    split_params = []
    for l in layer_groups:
        l1,l2 = [],[]
        for c in l.children():
            if isinstance(c, bn_types): l2 += list(trainable_params(c))
            elif isinstance(c, bias_types):
                bias = c.bias if hasattr(c, 'bias') else None
                l1 += [p for p in trainable_params(c) if not (p is bias)]
                if bias is not None: l2.append(bias)
            else: l1 += list(trainable_params(c))
        #Since we scan the children separately, we might get duplicates (tied weights). We need to preserve the order
        #for the optimizer load of state_dict
        l1,l2 = uniqueify(l1),uniqueify(l2)
        split_params += [l1, l2]      
    return split_params

def set_bn_eval(m:nn.Module)->None:
    "Set bn layers in eval mode for all recursive children of `m`."
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            l.eval()
        set_bn_eval(l)

def batch_to_half(b:Collection[Tensor])->Collection[Tensor]:
    "Set the input of batch `b` to half precision."
    return [to_half(b[0]), b[1]]

def bn2float(module:nn.Module)->nn.Module:
    "If `module` is batchnorm don't use half precision."
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:nn.Module)->nn.Module:
    "Convert `model` to half precision except the batchnorm layers."
    return bn2float(model.half())

def init_default(m:nn.Module, func:LayerFunc=nn.init.kaiming_normal_)->None:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m

def cond_init(m:nn.Module, init_func:LayerFunc):
    "Initialize the non-batchnorm layers of `m` with `init_func`."
    if (not isinstance(m, bn_types)) and requires_grad(m): init_default(m, init_func)

def apply_leaf(m:nn.Module, f:LayerFunc):
    "Apply `f` to children of `m`."
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)

def apply_init(m, init_func:LayerFunc):
    "Initialize all non-batchnorm layers of `m` with `init_func`."
    apply_leaf(m, partial(cond_init, init_func=init_func))

def in_channels(m:nn.Module) -> List[int]:
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')

def calc_loss(y_pred:Tensor, y_true:Tensor, loss_func:LossFunction):
    "Calculate loss between `y_pred` and `y_true` using `loss_func`."
    if hasattr(loss_func, 'reduction'):
        old_red = getattr(loss_func, 'reduction')
        setattr(loss_func, 'reduction', 'none')
        l = loss_func(y_pred, y_true)
        setattr(loss_func, 'reduction', old_red)
        return l
    else: return loss_func(y_pred, y_true, reduction='none')

def model_type(dtype):
    "Return the torch type corresponding to `dtype`."
    return (torch.float32 if np.issubdtype(dtype, np.floating) else
            torch.int64 if np.issubdtype(dtype, np.integer)
            else None)

def np2model_tensor(a):
    "Tranform numpy array `a` to a tensor of the same type."
    dtype = model_type(a.dtype)
    res = as_tensor(a)
    if not dtype: return res
    return res.type(dtype)

def _pca(x, k=2):
    "Compute PCA of `x` with `k` dimensions."
    x = x-torch.mean(x,0)
    U,S,V = torch.svd(x.t())
    return torch.mm(x,U[:,:k])
torch.Tensor.pca = _pca

def trange_of(x): 
    "Create a tensor from `range_of(x)`."
    return torch.arange(len(x))

def to_np(x): 
    "Convert a tensor to a numpy array."
    return x.data.cpu().numpy()

# monkey patching to allow matplotlib to plot tensors
def tensor__array__(self, dtype=None):
    res = to_np(self)
    if dtype is None: return res
    else: return res.astype(dtype, copy=False)
Tensor.__array__ = tensor__array__
Tensor.ndim = property(lambda x: len(x.shape))

def grab_idx(x,i,batch_first:bool=True):
    "Grab the `i`-th batch in `x`, `batch_first` stating the batch dimension."
    if batch_first: return ([o[i].cpu() for o in x]   if is_listy(x) else x[i].cpu())
    else:           return ([o[:,i].cpu() for o in x] if is_listy(x) else x[:,i].cpu())

def logit(x:Tensor)->Tensor:
    "Logit of `x`, clamped to avoid inf."
    x = x.clamp(1e-7, 1-1e-7)
    return -(1/x-1).log()

def logit_(x:Tensor)->Tensor:
    "Inplace logit of `x`, clamped to avoid inf"
    x.clamp_(1e-7, 1-1e-7)
    return (x.reciprocal_().sub_(1)).log_().neg_()

def uniform(low:Number, high:Number=None, size:Optional[List[int]]=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=`low`, max=`high`."
    if high is None: high=low
    return random.uniform(low,high) if size is None else torch.FloatTensor(*listify(size)).uniform_(low,high)

def log_uniform(low, high, size:Optional[List[int]]=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=log(`low`), max=log(`high`)."
    res = uniform(log(low), log(high), size)
    return exp(res) if size is None else res.exp_()

def rand_bool(p:float, size:Optional[List[int]]=None)->BoolOrTensor:
    "Draw 1 or shape=`size` random booleans (`True` occuring with probability `p`)."
    return uniform(0,1,size)<p

def uniform_int(low:int, high:int, size:Optional[List[int]]=None)->IntOrTensor:
    "Generate int or tensor `size` of ints between `low` and `high` (included)."
    return random.randint(low,high) if size is None else torch.randint(low,high+1,size)

def one_param(m: nn.Module)->Tensor: 
    "Return the first parameter of `m`."
    return next(m.parameters())

def try_int(o:Any)->Any:
    "Try to convert `o` to int, default to `o` if not possible."
    # NB: single-item rank-1 array/tensor can be converted to int, but we don't want to do this
    if isinstance(o, (np.ndarray,Tensor)): return o if o.ndim else int(o)
    if isinstance(o, collections.Sized) or getattr(o,'__array_interface__',False): return o
    try: return int(o)
    except: return o

def get_model(model:nn.Module):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, nn.DataParallel) else model

def flatten_check(out:Tensor, targ:Tensor) -> Tensor:
    "Check that `out` and `targ` have the same number of elements and flatten them."
    out,targ = out.contiguous().view(-1),targ.contiguous().view(-1)
    assert len(out) == len(targ), f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
    return out,targ

#Monkey-patch nn.DataParallel.reset
def _data_parallel_reset(self): 
    if hasattr(self.module, 'reset'): self.module.reset()
nn.DataParallel.reset = _data_parallel_reset
