from .imports import *
from .torch_imports import *

def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

conv_dict = {np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
    np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}

def T(a):
    if torch.is_tensor(a): res = a
    else:
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            return torch.FloatTensor(a.astype(np.float32))
        else: raise NotImplementedError
    return to_gpu(res, async=True)

def create_variable(x, volatile, requires_grad=False):
    if not isinstance(x, Variable):
        x = Variable(T(x), volatile=volatile, requires_grad=requires_grad)
    return x

def V_(x, requires_grad=False):
    return create_variable(x, False, requires_grad=requires_grad)
def V(x, requires_grad=False):
    return [V_(o, requires_grad) for o in x] if isinstance(x,list) else V_(x, requires_grad)

def VV_(x): return create_variable(x, True)
def VV(x):  return [VV_(o) for o in x] if isinstance(x,list) else VV_(x)

def to_np(v):
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    return v.cpu().numpy()

USE_GPU=True
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def noop(*args, **kwargs): return

def split_by_idxs(seq, idxs):
    last, sl = 0, len(seq)
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]

def chain_params(p):
    if isinstance(p, (list,tuple)):
        return list(chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def SGD_Momentum(momentum):
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def one_hot(a,c): return np.eye(c)[a]
