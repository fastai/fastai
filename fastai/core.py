from .imports import *
from .torch_imports import *

def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

def is_listy(x): return isinstance(x, (list,tuple))
def is_iter(x): return isinstance(x, collections.Iterable)
def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)

conv_dict = {np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
    np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}

def A(*a):
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def T_(a):
    if torch.is_tensor(a): return a
    a = np.array(np.ascontiguousarray(a))
    if a.dtype in (np.int8, np.int16, np.int32, np.int64):
        return torch.LongTensor(a.astype(np.int64))
    if a.dtype in (np.float32, np.float64):
        return torch.FloatTensor(a.astype(np.float32))
    raise NotImplementedError(a.dtype)
def T(a): return to_gpu(T_(a), async=True)

def create_variable(x, volatile, requires_grad=False):
    if not isinstance(x, Variable):
        x = Variable(T(x), volatile=volatile, requires_grad=requires_grad)
    return x

def V_(x, requires_grad=False, volatile=False): return create_variable(x, volatile, requires_grad)
def VV_(x):                                     return create_variable(x, True)
def V(x, requires_grad=False, volatile=False): return map_over(x, lambda o: V_(o, requires_grad, volatile))
def VV(x):                                     return map_over(x, VV_)

def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    return v.cpu().numpy()

USE_GPU=True
def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if torch.cuda.is_available() and USE_GPU else x

def noop(*args, **kwargs): return

def split_by_idxs(seq, idxs):
    last = 0
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def trainable_params_(m):
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

def partition(a, sz): return [a[i:i+sz] for i in range(0, len(a), sz)]

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


def save(fn, a): pickle.dump(a, open(fn,'wb'))
def load(fn): return pickle.load(open(fn,'rb'))
def load2(fn): return pickle.load(open(fn,'rb'), encoding='iso-8859-1')

def load_array(fname): return bcolz.open(fname)[:]


def chunk_iter(iterable, chunk_size):
    while True:
        chunk = []
        try:
            for _ in range(chunk_size): chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break

