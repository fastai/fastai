"Callbacks provides extensibility to the `basic_train` loop. See `train` for examples of custom callbacks."
from .basic_data import *
from .torch_core import *

__all__ = ['AverageMetric', 'Callback', 'CallbackHandler', 'OptimWrapper', 'SmoothenValue', 'Stepper', 'annealing_cos', 'CallbackList',
           'annealing_exp', 'annealing_linear', 'annealing_no', 'annealing_poly']

class OptimWrapper():
    "Basic wrapper around `opt` to simplify hyper-parameters changes."
    def __init__(self, opt:optim.Optimizer, wd:Floats=0., true_wd:bool=False, bn_wd:bool=True):
        self.opt,self.true_wd,self.bn_wd = opt,true_wd,bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func:Union[type,Callable], lr:Union[float,Tuple,List],
               layer_groups:ModuleList, **kwargs:Any)->optim.Optimizer:
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr':0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr,opt.opt_func = listify(lr, layer_groups),opt_func
        return opt
    
    def new(self, layer_groups:ModuleList):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr':0} for l in split_groups])
        return self.create(opt_func, self.lr, layer_groups, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def __repr__(self)->str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    #Pytorch optimizer methods
    def step(self)->None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr,wd,pg1,pg2 in zip(self._lr,self._wd,self.opt.param_groups[::2],self.opt.param_groups[1::2]):
                for p in pg1['params']: p.data.mul_(1 - wd*lr)
                if self.bn_wd:
                    for p in pg2['params']: p.data.mul_(1 - wd*lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self)->None:
        "Clear optimizer gradients."
        self.opt.zero_grad()
        
    #Passthrough to the inner opt.
    def __getattr__(self,k:str)->Any: return getattr(self.opt, k, None)
    
    def clear(self):
        "Reset the state of the inner optimizer."
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    #Hyperparameters as properties
    @property
    def lr(self)->float: return self._lr[-1]
    @lr.setter
    def lr(self, val:float)->None:
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self)->float:return self._mom[-1]
    @mom.setter
    def mom(self, val:float)->None:
        if 'momentum' in self.opt_keys: self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:  self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self)->float: return None if self._beta is None else self._beta[-1]
    @beta.setter
    def beta(self, val:float)->None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:    self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:  self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self)->float: return self._wd[-1]
    @wd.setter
    def wd(self, val:float)->None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    #Helper functions
    def read_defaults(self)->None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom,self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key:str, val:Any, bn_groups:bool=True)->Any:
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val): val = [(v1,v2) for v1,v2 in zip(*val)]
        for v,pg1,pg2 in zip(val,self.opt.param_groups[::2],self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key:str) -> Union[List[float],Tuple[List[float],List[float]]]:
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val

class Callback():
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order=0
    def on_train_begin(self, **kwargs:Any)->None:
        "To initialize constants in the callback."
        pass
    def on_epoch_begin(self, **kwargs:Any)->None:
        "At the beginning of each epoch."
        pass
    def on_batch_begin(self, **kwargs:Any)->None:
        "Set HP before the step is done. Returns xb, yb (which can allow us to modify the input at that step if needed)."
        pass
    def on_loss_begin(self, **kwargs:Any)->None:
        "Called after forward pass but before loss has been computed. Returns the output (which can allow us to modify it)."
        pass
    def on_backward_begin(self, **kwargs:Any)->None:
        """Called after the forward pass and the loss has been computed, but before backprop.
           Returns the loss (which can allow us to modify it, for instance for reg functions)"""
        pass
    def on_backward_end(self, **kwargs:Any)->None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass
    def on_step_end(self, **kwargs:Any)->None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass
    def on_batch_end(self, **kwargs:Any)->None:
        "Called at the end of the batch."
        pass
    def on_epoch_end(self, **kwargs:Any)->bool:
        "Called at the end of an epoch."
        return False
    def on_train_end(self, **kwargs:Any)->None:
        "Useful for cleaning up things and saving files/models."
        pass

class SmoothenValue():
    "Create a smooth moving average for a value (loss, etc) using `beta`."
    def __init__(self, beta:float):
        self.beta,self.n,self.mov_avg = beta,0,0

    def add_value(self, val:float)->None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)

CallbackList = Collection[Callback]

def _get_init_state(): return {'epoch':0, 'iteration':0, 'num_batch':0}

@dataclass
class CallbackHandler():
    "Manage all of the registered `callbacks` and `metrics`, smoothing loss by momentum `beta`."
    callbacks:CallbackList=None
    metrics:CallbackList=None
    beta:float=0.98

    def __post_init__(self)->None:
        "Initialize smoother and learning stats."
        self.callbacks = ifnone(self.callbacks, [])
        self.metrics = ifnone(self.metrics, [])
        self.metrics = [(met if isinstance(met, Callback) else AverageMetric(met)) for met in self.metrics]
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict:Dict[str,Union[int,float,Tensor]]=_get_init_state()

    def __call__(self, cb_name, call_mets=True, **kwargs)->None:
        "Call through to all of the `CallbakHandler` functions."
        if call_mets: [getattr(met, f'on_{cb_name}')(**self.state_dict, **kwargs) for met in self.metrics]
        return [getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs) for cb in self.callbacks]

    def on_train_begin(self, epochs:int, pbar:PBar, metrics:MetricFuncList)->None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict['n_epochs'],self.state_dict['pbar'],self.state_dict['metrics'] = epochs,pbar,metrics
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)

    def on_epoch_begin(self)->None:
        "Handle new epoch."
        self.state_dict['num_batch'] = 0
        self('epoch_begin')

    def on_batch_begin(self, xb:Tensor, yb:Tensor, train:bool=True)->None:
        "Handle new batch `xb`,`yb` in `train` or validation."
        self.state_dict['last_input'], self.state_dict['last_target'] = xb, yb
        self.state_dict['train'] = train
        cbs = self.callbacks if train else self.metrics + self.callbacks
        for cb in self.callbacks:
            a = cb.on_batch_begin(**self.state_dict)
            if a is not None: self.state_dict['last_input'], self.state_dict['last_target'] = a
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out:Tensor)->None:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        for cb in self.callbacks:
            a = cb.on_loss_begin(**self.state_dict)
            if a is not None: self.state_dict['last_output'] = a
        return self.state_dict['last_output']

    def on_backward_begin(self, loss:Tensor)->None:
        "Handle gradient calculation on `loss`."
        self.smoothener.add_value(loss.detach().cpu())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        for cb in self.callbacks:
            a = cb.on_backward_begin(**self.state_dict)
            if a is not None: self.state_dict['last_loss'] = a
        return self.state_dict['last_loss']

    def on_backward_end(self)->None:
        "Handle end of gradient calculation."
        self('backward_end', False)
    def on_step_end(self)->None:
        "Handle end of optimization step."
        self('step_end', False)

    def on_batch_end(self, loss:Tensor)->None:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        stop = np.any(self('batch_end', not self.state_dict['train']))
        if self.state_dict['train']:
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return stop

    def on_epoch_end(self, val_loss:Tensor)->bool:
        "Epoch is done, process `val_loss`."
        self.state_dict['last_metrics'] = [val_loss] if val_loss is not None else None
        self.state_dict['epoch'] += 1
        if not self.state_dict['train']:
            for met in self.metrics:
                met.on_epoch_end(**self.state_dict)
                self.state_dict['last_metrics'].append(met.metric)
        return np.any(self('epoch_end', False))

    def on_train_end(self, exception:Union[bool,Exception])->None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)

class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func,'func',func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        self.count += last_target[0].size(0)
        self.val += last_target[0].size(0) * self.func(last_output, *last_target).detach().cpu()

    def on_epoch_end(self, **kwargs):
        "Sets the final result in `self.metric`."
        self.metric = self.val/self.count

def annealing_no(start:Number, end:Number, pct:float)->Number:
    "No annealing, always return `start`."
    return start
def annealing_linear(start:Number, end:Number, pct:float)->Number:
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
def annealing_exp(start:Number, end:Number, pct:float)->Number:
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct
def annealing_cos(start:Number, end:Number, pct:float)->Number:
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

def do_annealing_poly(start:Number, end:Number, pct:float, degree:Number)->Number:
    "Helper function for `anneal_poly`."
    return end + (start-end) * (1-pct)**degree
def annealing_poly(degree:Number)->Number:
    "Anneal polynomically from `start` to `end` as pct goes from 0.0 to 1.0."
    return functools.partial(do_annealing_poly, degree=degree)

class Stepper():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"
    def __init__(self, vals:StartOptEnd, n_iter:int, func:Optional[AnnealFunc]=None):
        self.start,self.end = (vals[0],vals[1]) if is_tuple(vals) else (vals,0)
        self.n_iter = max(1,n_iter)
        if func is None: self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:          self.func = func
        self.n = 0

    def step(self)->Number:
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self)->bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter

