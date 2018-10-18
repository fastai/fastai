"Callbacks provides extensibility to the `basic_train` loop. See `train` for examples of custom callbacks."
from .data import *
from .torch_core import *

__all__ = ['Callback', 'CallbackHandler', 'OptimWrapper', 'SmoothenValue', 'Stepper', 'annealing_cos', 'CallbackList',
           'annealing_exp', 'annealing_linear', 'annealing_no', 'annealing_poly', 'do_annealing_poly']

class OptimWrapper():
    "Basic wrapper around an optimizer to simplify HP changes."
    def __init__(self, opt:optim.Optimizer, wd:Floats=0., true_wd:bool=False, bn_wd:bool=True):
        self.opt,self.true_wd,self.bn_wd = opt,true_wd,bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_fn:Union[type,Callable], lr:Union[float,Tuple,List],
               layer_groups:ModuleList, **kwargs:Any)->optim.Optimizer:
        "Create an optim.Optimizer from `opt_fn` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_fn([{'params': trainable_params(l), 'lr':0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr = listify(lr, layer_groups)
        return opt

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

    #Hyperparameters as properties
    @property
    def lr(self)->float:
        "Get learning rate."
        return self._lr[-1]

    @lr.setter
    def lr(self, val:float)->None:
        "Set learning rate."
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self)->float:
        "Get momentum."
        return self._mom[-1]

    @mom.setter
    def mom(self, val:float)->None:
        "Set momentum."
        if 'momentum' in self.opt_keys: self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:  self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self)->float:
        "Get beta (or alpha as makes sense for given optimizer)."
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val:float)->None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:    self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:  self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self)->float:
        "Get weight decay."
        return self._wd[-1]

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
        "Set the values inside the optimizer dictionary at the key."
        if is_tuple(val): val = [(v1,v2) for v1,v2 in zip(*val)]
        for v,pg1,pg2 in zip(val,self.opt.param_groups[::2],self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key:str) -> Union[List[float],Tuple[List[float],List[float]]]:
        "Read a hyperparameter key in the optimizer dictionary."
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
    "Create a smooth moving average for a value (loss, etc)."
    def __init__(self, beta:float):
        "Create smoother for value, beta should be 0<beta<1."
        self.beta,self.n,self.mov_avg = beta,0,0

    def add_value(self, val:float)->None:
        "Add current value to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)

CallbackList = Collection[Callback]

def _get_init_state(): return {'epoch':0, 'iteration':0, 'num_batch':0}

@dataclass
class CallbackHandler():
    "Manage all of the registered callback objects, smoothing loss by momentum `beta`."
    callbacks:CallbackList
    beta:float=0.98

    def __post_init__(self)->None:
        "Initialize smoother and learning stats."
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict:Dict[str,Union[int,float,Tensor]]=_get_init_state()

    def __call__(self, cb_name, **kwargs)->None:
        "Call through to all of the `CallbakHandler` functions."
        return [getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs) for cb in self.callbacks]

    def on_train_begin(self, epochs:int, pbar:PBar, metrics:MetricFuncList)->None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict['n_epochs'],self.state_dict['pbar'],self.state_dict['metrics'] = epochs,pbar,metrics
        self('train_begin')

    def on_epoch_begin(self)->None:
        "Handle new epoch."
        self.state_dict['num_batch'] = 0
        self('epoch_begin')

    def on_batch_begin(self, xb:Tensor, yb:Tensor, train:bool=True)->None:
        "Handle new batch `xb`,`yb`."
        self.state_dict['last_input'], self.state_dict['last_target'] = xb, yb
        self.state_dict['train'] = train
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
        self.smoothener.add_value(loss.detach())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        for cb in self.callbacks:
            a = cb.on_backward_begin(**self.state_dict)
            if a is not None: self.state_dict['last_loss'] = a
        return self.state_dict['last_loss']

    def on_backward_end(self)->None:
        "Handle end of gradient calculation."
        self('backward_end')
    def on_step_end(self)->None:
        "Handle end of optimization step."
        self('step_end')

    def on_batch_end(self, loss:Tensor)->None:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        stop = np.any(self('batch_end'))
        if self.state_dict['train']: 
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return stop

    def on_epoch_end(self, val_metrics:MetricsList)->bool:
        "Epoch is done, process `val_metrics`."
        self.state_dict['last_metrics'] = val_metrics
        stop = np.any(self('epoch_end'))
        self.state_dict['epoch'] += 1
        return stop

    def on_train_end(self, exception:Union[bool,Exception])->None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)

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
        self.n_iter = n_iter
        if func is None: self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:          self.func = func
        self.n = 0

    def step(self)->Number:
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self)->bool:
        "Schedule completed."
        return self.n >= self.n_iter

