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
            res = torch.FloatTensor(a.astype(np.float32))
        else: raise NotImplementedError(a.dtype)
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
    last = 0
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
