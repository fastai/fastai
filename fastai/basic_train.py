"Provides basic training and validation with `Learner`"
from .torch_core import *
from .basic_data import *
from .callback import *

__all__ = ['Learner', 'LearnerCallback', 'Recorder', 'RecordOnCPU', 'fit', 'loss_batch', 'train_epoch', 'validate',
           'get_preds']

defaults.lr = slice(3e-3)
defaults.wd = 1e-2

def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
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

    return loss.detach().cpu()

def get_preds(model:nn.Module, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None,
              activ:nn.Module=None, loss_func:OptLossFunc=None, n_batch:Optional[int]=None) -> List[Tensor]:
    "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
    res = [torch.cat(o).cpu() for o in
           zip(*validate(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
    if loss_func is not None: res.append(calc_loss(res[0], res[1], loss_func))
    if activ is not None: res[0] = activ(res[0])
    return res

def validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_losses.append(loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler))
            if not is_listy(yb): yb = [yb]
            nums.append(yb[0].shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses

def train_epoch(model:nn.Module, dl:DataLoader, opt:optim.Optimizer, loss_func:LossFunction)->None:
    "Simple training of `model` for 1 epoch of `dl` using optim `opt` and loss function `loss_func`."
    model.train()
    for xb,yb in dl:
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

def fit(epochs:int, model:nn.Module, loss_func:LossFunction, opt:optim.Optimizer,
        data:DataBunch, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None)->None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
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
                loss = loss_batch(model, xb, yb, loss_func, opt, cb_handler)
                if cb_handler.on_batch_end(loss): break

            if hasattr(data,'valid_dl') and data.valid_dl is not None and data.valid_ds is not None:
                val_loss = validate(model, data.valid_dl, loss_func=loss_func,
                                       cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise e
    finally: cb_handler.on_train_end(exception)

loss_func_name2activ = {'cross_entropy_loss': partial(F.softmax, dim=-1), 'nll_loss': torch.exp, 'poisson_nll_loss': torch.exp,
    'kl_div_loss': torch.exp, 'bce_with_logits_loss': torch.sigmoid, 'cross_entropy': partial(F.softmax, dim=1),
    'kl_div': torch.exp, 'binary_cross_entropy_with_logits': torch.sigmoid,
}

def _loss_func2activ(loss_func):
    if getattr(loss_func,'keywords',None):
        if not loss_func.keywords.get('log_input', True): return
    # flattened loss
    loss_func = getattr(loss_func, 'func', loss_func)
    # could have a partial inside flattened loss!
    loss_func = getattr(loss_func, 'func', loss_func)
    cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name == 'mix_up_loss':
        loss_func = loss_func.crit
        cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name in loss_func_name2activ:
        if cls_name == 'poisson_nll_loss' and (not getattr(loss_func, 'log_input', True)): return
        return loss_func_name2activ[cls_name]
    if getattr(loss_func,'__name__','') in loss_func_name2activ:
        return loss_func_name2activ[loss_func.__name__]
    return noop

@dataclass
class Learner():
    "Trainer for `model` using `data` to minimize `loss_func` with optimizer `opt_func`."
    data:DataBunch
    model:nn.Module
    opt_func:Callable=AdamW
    loss_func:Callable=None
    metrics:Collection[Callable]=None
    true_wd:bool=True
    bn_wd:bool=True
    wd:Floats=defaults.wd
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
        "Build differential learning rates from `lr`."
        if not isinstance(lr,slice): return lr
        if lr.start: res = even_mults(lr.start, lr.stop, len(self.layer_groups))
        else: res = [lr.stop/10]*(len(self.layer_groups)-1) + [lr.stop]
        return np.array(res)

    def fit(self, epochs:int, lr:Union[Floats,slice]=defaults.lr,
            wd:Floats=None, callbacks:Collection[Callback]=None)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        if not getattr(self, 'opt', False): self.create_opt(lr, wd)
        else: self.opt.lr,self.opt.wd = lr,wd
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
        self.create_opt(defaults.lr)

    def freeze(self)->None:
        "Freeze up to last layer."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)
        self.create_opt(defaults.lr)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)
        self.create_opt(defaults.lr)

    def __del__(self): del(self.model, self.data)

    def save(self, name:PathOrStr, return_path:bool=False, with_opt:bool=True):
        "Save model and optimizer state (if `with_opt`) with `name` to `self.model_dir`."
        path = self.path/self.model_dir/f'{name}.pth'
        if not with_opt: state = self.model.state_dict()
        else: state = {'model': self.model.state_dict(), 'opt':self.opt.state_dict()}
        torch.save(state, path)
        if return_path: return path

    def dl(self, ds_type:DatasetType=DatasetType.Valid):
        "Return DataLoader for DatasetType `ds_type`."
        return self.data.dl(ds_type)

    def load(self, name:PathOrStr, device:torch.device=None, strict:bool=True, with_opt:bool=None):
        "Load model and optimizer state (if `with_opt`) `name` from `self.model_dir` using `device`."
        if device is None: device = self.data.device
        state = torch.load(self.path/self.model_dir/f'{name}.pth', map_location=device)
        if set(state.keys()) == {'model', 'opt'}:
            self.model.load_state_dict(state['model'], strict=strict)
            if ifnone(with_opt,True):
                if not hasattr(self, 'opt'): opt = self.create_opt(defaults.lr, self.wd)
                try:    self.opt.load_state_dict(state['opt'])
                except: pass
        else:
            if with_opt: warn("Saved filed doesn't contain an optimizer state.")
            self.model.load_state_dict(state, strict=strict)
        return self

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None,
                  pbar:Optional[PBar]=None) -> List[Tensor]:
        "Return predictions and targets on `ds_type` dataset."
        lf = self.loss_func if with_loss else None
        return get_preds(self.model, self.dl(ds_type), cb_handler=CallbackHandler(self.callbacks),
                         activ=_loss_func2activ(self.loss_func), loss_func=lf, n_batch=n_batch, pbar=pbar)

    def pred_batch(self, ds_type:DatasetType=DatasetType.Valid, batch:Tuple=None) -> List[Tensor]:
        "Return output of the model on one batch from `ds_type` dataset."
        if batch: xb,yb = batch
        else: xb,yb = self.data.one_batch(ds_type, detach=False, denorm=False)
        cb_handler = CallbackHandler(self.callbacks)
        cb_handler.on_batch_begin(xb,yb, train=False)
        preds = loss_batch(self.model.eval(), xb, yb, cb_handler=cb_handler)
        return _loss_func2activ(self.loss_func)(preds[0])

    def backward(self, item):
        "Pass `item` through the model and computes the gradient. Useful if `backward_hooks` are attached."
        xb,yb = self.data.one_item(item)
        loss = loss_batch(self.model.eval(), xb, yb, self.loss_func, opt=FakeOptimizer(),
                          cb_handler=CallbackHandler(self.callbacks))
        return loss

    def predict(self, item:ItemBase, **kwargs):
        "Return prect class, label and probabilities for `item`."
        self.callbacks.append(RecordOnCPU())
        batch = self.data.one_item(item)
        res = self.pred_batch(batch=batch)
        pred = res[0]
        x = self.callbacks[-1].input
        norm = getattr(self.data,'norm',False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get('do_y',False): pred = self.data.denorm(pred)
        self.callbacks = self.callbacks[:-1]
        ds = self.data.single_ds
        pred = ds.y.analyze_pred(pred, **kwargs)
        out = ds.y.reconstruct(pred, ds.x.reconstruct(x[0])) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
        return out, pred, res[0]

    def validate(self, dl=None, callbacks=None, metrics=None):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        cb_handler.on_epoch_begin()
        val_metrics = validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']

    def show_results(self, ds_type=DatasetType.Valid, rows:int=5, **kwargs):
        "Show `rows` result of predictions on `ds_type` dataset."
        #TODO: get read of has_arg x and split_kwargs_by_func if possible
        ds = self.dl(ds_type).dataset
        self.callbacks.append(RecordOnCPU())
        preds = self.pred_batch(ds_type)
        *self.callbacks,rec_cpu = self.callbacks
        x,y = rec_cpu.input,rec_cpu.target
        norm = getattr(self.data,'norm',False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get('do_y',True):
                y     = self.data.denorm(y)
                preds = self.data.denorm(preds)
        analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(rows)]
        xs = [ds.x.reconstruct(grab_idx(x, i, self.data._batch_first)) for i in range(rows)]
        if has_arg(ds.y.reconstruct, 'x'):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
        else :
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(rows)]
            zs = [ds.y.reconstruct(z) for z in preds]
        ds.x.show_xyzs(xs, ys, zs, **kwargs)

class RecordOnCPU(Callback):
    "Store the `input` and `target` going through the model on the CPU."
    def on_batch_begin(self, last_input,last_target,**kwargs):
        self.input,self.target = to_cpu(last_input),to_cpu(last_target)

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
        self.no_val = False

    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        "Initialize recording status at beginning of training."
        self.pbar = pbar
        self.names = ['epoch', 'train_loss'] if self.no_val else ['epoch', 'train_loss', 'valid_loss']
        self.names += metrics_names
        if hasattr(self, '_added_met_names'): self.names += self._added_met_names
        self.pbar.write(self.names, table=True)
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
        else: last_metrics = [] if self.no_val else [None]
        if hasattr(self, '_added_mets'): last_metrics += self._added_mets
        if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
        self.format_stats([epoch, smooth_loss] + last_metrics)
        return False

    def format_stats(self, stats:TensorOrNumList)->None:
        "Format stats before printing."
        str_stats = []
        for name,stat in zip(self.names,stats):
            str_stats.append('' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.6f}')
        self.pbar.write(str_stats, table=True)

    def add_metrics(self, metrics):
        "Add `metrics` to the inner stats."
        self._added_mets = metrics

    def add_metric_names(self, names):
        "Add `names` to the inner metric names."
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
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    def plot_losses(self, last:int=None)->None:
        "Plot training and validation losses."
        last = ifnone(last,len(self.nb_batches))
        assert last<=len(self.nb_batches), f"We can only plot up to the last {len(self.nb_batches)} epochs. Please adapt 'last' parameter accordingly."
        _, ax = plt.subplots(1,1)
        l_b = np.sum(self.nb_batches[-last:])
        iterations = range_of(self.losses)[-l_b:]
        ax.plot(iterations, self.losses[-l_b:], label='Train')
        val_iter = self.nb_batches[-last:]
        val_iter = np.cumsum(val_iter)+np.sum(self.nb_batches[:-last])
        ax.plot(val_iter, self.val_losses[-last:], label='Validation')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Batches processed')
        ax.legend()

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

class FakeOptimizer():
    def step(self): pass
    def zero_grad(self): pass

