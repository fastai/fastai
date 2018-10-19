"Provides basic training and validation with `Learner`"
from .torch_core import *
from .basic_data import *
from .callback import *

__all__ = ['Learner', 'LearnerCallback', 'Recorder', 'fit', 'loss_batch', 'train_epoch', 'validate',
           'get_preds', 'default_lr', 'default_wd']

default_lr = slice(3e-3)
default_wd = 1e-2

def loss_batch(model:Model, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None, 
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler([], []))
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return (loss.detach().cpu(),)

def get_preds(model:Model, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None) -> List[Tensor]:
    "Predict the output of the elements in the dataloader."
    return [torch.cat(o).cpu() for o in zip(*validate(model, dl, pbar=pbar, cb_handler=cb_handler, average=False))]

def validate(model:Model, dl:DataLoader, loss_func:OptLossFunc=None,
             cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average=True)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate loss and metrics for the validation set."
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_losses.append(loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler))
            if not is_listy(yb): yb = [yb]
            nums.append(yb[0].shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[0]): break
        nums = np.array(nums, dtype=np.float32)
        if average: return [(to_np(torch.stack(val)) * nums).sum() / nums.sum() for val in zip(*val_losses)]
        else:       return val_losses

def train_epoch(model:Model, dl:DataLoader, opt:optim.Optimizer, loss_func:LossFunction)->None:
    "Simple training of `model` for 1 epoch of `dl` using optim `opt` and loss function `loss_func`."
    model.train()
    for xb,yb in dl:
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

def fit(epochs:int, model:Model, loss_func:LossFunction, opt:optim.Optimizer,
        data:DataBunch, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None)->None:
    "Fit the `model` on `data` and learn using `loss` and `opt`."
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            model.train()
            cb_handler.on_epoch_begin()

            for xb,yb in progress_bar(data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(model, xb, yb, loss_func, opt, cb_handler)[0]
                if cb_handler.on_batch_end(loss): break

            if hasattr(data,'valid_dl') and data.valid_dl is not None:
                val_loss = validate(model, data.valid_dl, loss_func=loss_func,
                                       cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise e
    finally: cb_handler.on_train_end(exception)

@dataclass
class Learner():
    "Train `model` using `data` to minimize `loss_func` with optimizer `opt_func`."
    data:DataBunch
    model:nn.Module
    opt_func:Callable=AdamW
    loss_func:Callable=None
    metrics:Collection[Callable]=None
    true_wd:bool=True
    bn_wd:bool=True
    wd:Floats=default_wd
    train_bn:bool=True
    path:str = None
    model_dir:str = 'models'
    callback_fns:Collection[Callable]=None
    callbacks:Collection[Callback]=field(default_factory=list)
    layer_groups:Collection[nn.Module]=None
    def __post_init__(self)->None:
        "Setup path,metrics, callbacks and ensure model directory exists."
        self.path = Path(ifnone(self.path, self.data.path))
        (self.path/self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model = self.model.to(self.data.device)
        self.loss_func = ifnone(self.loss_func, self.data.loss_func)
        self.metrics=listify(self.metrics)
        if not self.layer_groups: self.layer_groups = [nn.Sequential(*flatten_model(self.model))]
        self.callbacks = listify(self.callbacks)
        self.callback_fns = [Recorder] + listify(self.callback_fns)

    def init(self, init): apply_init(self.model, init)

    def lr_range(self, lr:Union[float,slice])->np.ndarray:
        "Build differential learning rates."
        if not isinstance(lr,slice): return lr
        if lr.start: res = even_mults(lr.start, lr.stop, len(self.layer_groups))
        else: res = [lr.stop/3]*(len(self.layer_groups)-1) + [lr.stop]
        return np.array(res)

    def fit(self, epochs:int, lr:Union[Floats,slice]=default_lr,
            wd:Floats=None, callbacks:Collection[Callback]=None)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        self.create_opt(lr, wd)
        callbacks = [cb(self) for cb in self.callback_fns] + listify(callbacks)
        fit(epochs, self.model, self.loss_func, opt=self.opt, data=self.data, metrics=self.metrics,
            callbacks=self.callbacks+callbacks)

    def create_opt(self, lr:Floats, wd:Floats=0.)->None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = OptimWrapper.create(self.opt_func, lr, self.layer_groups, wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)

    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self)->None:
        "Freeze up to last layer."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def __del__(self): del(self.model, self.data)

    def save(self, name:PathOrStr):
        "Save model with `name` to `self.model_dir`."
        torch.save(self.model.state_dict(), self.path/self.model_dir/f'{name}.pth')

    def load(self, name:PathOrStr):
        "Load model `name` from `self.model_dir`."
        self.model.load_state_dict(torch.load(self.path/self.model_dir/f'{name}.pth'))

    def get_preds(self, is_test:bool=False) -> List[Tensor]:
        "Return predictions and targets on the valid or test set, depending on `is_test`."
        return get_preds(self.model, self.data.holdout(is_test), cb_handler=CallbackHandler(self.callbacks, []))
    
    def validate(self, dl=None, callbacks=None, metrics=None):
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        cb_handler.on_epoch_begin()
        val_metrics = validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']

@dataclass
class LearnerCallback(Callback):
    "Base class for creating callbacks for a `Learner`."
    learn: Learner
    def __post_init__(self):
        if self.cb_name: setattr(self.learn, self.cb_name, self)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)

class Recorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order=-10
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.opt = self.learn.opt
        self.train_dl = self.learn.data.train_dl

    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        "Initialize recording status at beginning of training."
        self.pbar = pbar
        self.names = ['epoch', 'train loss', 'valid loss'] + metrics_names
        self.pbar.write('  '.join(self.names))
        self.losses,self.val_losses,self.lrs,self.moms,self.metrics,self.nb_batches = [],[],[],[],[],[]

    def on_batch_begin(self, train, **kwargs:Any)->None:
        "Record learning rate and momentum at beginning of batch."
        if train:
            self.lrs.append(self.opt.lr)
            self.moms.append(self.opt.mom)

    def on_backward_begin(self, smooth_loss:Tensor, **kwargs:Any)->None:
        "Record the loss before any other callback has a chance to modify it."
        self.losses.append(smooth_loss)
        if self.pbar is not None and hasattr(self.pbar,'child'):
            self.pbar.child.comment = f'{smooth_loss:.4f}'

    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:Tensor,
                     last_metrics=MetricsList, **kwargs:Any)->bool:
        "Save epoch info: num_batch, smooth_loss, metrics."
        self.nb_batches.append(num_batch)
        if last_metrics is not None:
            self.val_losses.append(last_metrics[0])
            if hasattr(self, '_added_mets'): last_metrics += self._added_mets
            if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
            self.format_stats([epoch, smooth_loss] + last_metrics)
        else:  self.format_stats([epoch, smooth_loss])
        return False

    def format_stats(self, stats:TensorOrNumList)->None:
        "Format stats before printing."
        str_stats = []
        for name,stat in zip(self.names,stats):
            t = str(stat) if isinstance(stat, int) else f'{stat:.6f}'
            t += ' ' * (len(name) - len(t))
            str_stats.append(t)
        self.pbar.write('  '.join(str_stats))
        
    def add_metrics(self, metrics):
        self._added_mets = metrics

    def add_metric_names(self, names):
        self._added_met_names = names
        
    def plot_lr(self, show_moms=False)->None:
        "Plot learning rate, `show_moms` to include momentum."
        iterations = range_of(self.lrs)
        if show_moms:
            _, axs = plt.subplots(1,2, figsize=(12,4))
            axs[0].plot(iterations, self.lrs)
            axs[1].plot(iterations, self.moms)
        else: plt.plot(iterations, self.lrs)

    def plot(self, skip_start:int=10, skip_end:int=5)->None:
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        _, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')

    def plot_losses(self)->None:
        "Plot training and validation losses."
        _, ax = plt.subplots(1,1)
        iterations = range_of(self.losses)
        ax.plot(iterations, self.losses)
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        ax.plot(val_iter, self.val_losses)

    def plot_metrics(self)->None:
        "Plot metrics collected during training."
        assert len(self.metrics) != 0, "There are no metrics to plot."
        _, axes = plt.subplots(len(self.metrics[0]),1,figsize=(6, 4*len(self.metrics[0])))
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]
        for i, ax in enumerate(axes):
            values = [met[i] for met in self.metrics]
            ax.plot(val_iter, values)
