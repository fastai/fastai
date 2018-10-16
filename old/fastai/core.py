from .imports import *
from .torch_imports import *

def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

def is_listy(x): return isinstance(x, (list,tuple))
def is_iter(x): return isinstance(x, collections.Iterable)
def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)
def map_none(x, f): return None if x is None else f(x)
def delistify(x): return x[0] if is_listy(x) else x
def listify(x, y):
    if not is_iter(x): x=[x]
    n = y if type(y)==int else len(y)
    if len(x)==1: x = x * n
    return x

def datafy(x):
    if is_listy(x): return [o.data for o in x]
    else:           return x.data

conv_dict = {np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
    np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor. 
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = to_half(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a)
    return a

def to_half(tensor):
    if torch.cuda.is_available():
        return torch.cuda.HalfTensor(tensor)
    else:
        return torch.FloatTensor(tensor)

def create_variable(x, volatile, requires_grad=False):
    if type (x) != Variable:
        if IS_TORCH_04: x = Variable(T(x), requires_grad=requires_grad)
        else:           x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x

def V_(x, requires_grad=False, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)
def V(x, requires_grad=False, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))

def VV_(x): 
    '''creates a volatile tensor, which does not require gradients. '''
    return create_variable(x, True)

def VV(x):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, VV_)

def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if torch.cuda.is_available():
        if is_half_tensor(v): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

def is_half_tensor(v):
    return isinstance(v, torch.cuda.HalfTensor)


IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()
def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x

def noop(*args, **kwargs): return

def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
          raise KeyError(f'Idx {idx} is out-of-bounds')
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def trainable_params_(m):
    '''Returns a list of trainable parameters in the model m. (i.e., those that require gradients.)'''
    return [p for p in m.parameters() if p.requires_grad]

def chain_params(p):
    if is_listy(p):
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

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a):
    return partition(a, len(a)//num_cpus() + 1)

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


class BasicModel():
    def __init__(self,model,name='unnamed'): self.model,self.name = model,name
    def get_layer_groups(self, do_fc=False): return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self): return [self.model]

class SimpleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)


def save(fn, a): 
    """Utility function that savess model, function, etc as pickle"""    
    pickle.dump(a, open(fn,'wb'))
def load(fn): 
    """Utility function that loads model, function, etc as pickle"""
    return pickle.load(open(fn,'rb'))
def load2(fn):
    """Utility function allowing model piclking across Python2 and Python3"""
    return pickle.load(open(fn,'rb'), encoding='iso-8859-1')

def load_array(fname): 
    '''
    Load array using bcolz, which is based on numpy, for fast array saving and loading operations. 
    https://github.com/Blosc/bcolz
    '''
    return bcolz.open(fname)[:]


def chunk_iter(iterable, chunk_size):
    '''A generator that yields chunks of iterable, chunk_size at a time. '''
    while True:
        chunk = []
        try:
            for _ in range(chunk_size): chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break

def set_grad_enabled(mode): return torch.set_grad_enabled(mode) if IS_TORCH_04 else contextlib.suppress()

def no_grad_context(): return torch.no_grad() if IS_TORCH_04 else contextlib.suppress()
