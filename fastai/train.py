"Provides advanced training extensions to `fastai.basic_train`. Includes half-precision, learning rate finder, mixup, and one-cycle"
from .torch_core import *
from .callbacks import *
from .basic_train import *

__all__ = ['BnFreeze', 'GradientClipping', 'ShowGraph', 'fit_one_cycle', 'lr_find', 'one_cycle_scheduler', 'to_fp16', 'mixup']

def one_cycle_scheduler(lr_max:float, **kwargs:Any)->OneCycleScheduler:
    return partial(OneCycleScheduler, lr_max=lr_max, **kwargs)

def fit_one_cycle(learn:Learner, cyc_len:int, max_lr:Union[Floats,slice]=default_lr, 
                  moms:Tuple[float,float]=(0.95,0.85), div_factor:float=25., pct_start:float=0.3, 
                  wd:float=None, callbacks:Optional[CallbackList]=None, **kwargs)->None:
    "Fit a model following the 1cycle policy."
    max_lr = learn.lr_range(max_lr)
    callbacks = ifnone(callbacks, [])
    callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms, div_factor=div_factor,
                                        pct_start=pct_start, **kwargs))
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)

def lr_find(learn:Learner, start_lr:Floats=1e-7, end_lr:Floats=10, num_it:int=100, **kwargs:Any):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`."
    start_lr = np.array(start_lr) if is_listy(start_lr) else start_lr
    end_lr = np.array(end_lr) if is_listy(end_lr) else end_lr
    cb = LRFinder(learn, start_lr, end_lr, num_it)
    a = int(np.ceil(num_it/len(learn.data.train_dl)))
    learn.fit(a, start_lr, callbacks=[cb], **kwargs)

def to_fp16(learn:Learner, loss_scale:float=512., flat_master:bool=False)->Learner:
    "Transform `learn` in FP16 precision."
    learn.model = model2half(learn.model)
    learn.mp_cb = MixedPrecision(learn, loss_scale=loss_scale, flat_master=flat_master)
    learn.callbacks.append(learn.mp_cb)
    return learn

def mixup(learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True) -> Learner:
    "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."
    if stack_y: learn.loss_func = MixUpLoss(learn.loss_func)
    learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))
    return learn

Learner.fit_one_cycle = fit_one_cycle
Learner.lr_find = lr_find
Learner.to_fp16 = to_fp16
Learner.mixup = mixup

class ShowGraph(LearnerCallback):
    "Update a graph of learner stats and metrics after each epoch."
    def on_epoch_end(self, n_epochs:int, last_metrics:MetricsList, **kwargs)->bool:
        "If we have metrics plot them in our pbar graph"
        if last_metrics is not None:
            rec = self.learn.recorder
            iters = list(range(len(rec.losses)))
            val_iter = np.array(rec.nb_batches).cumsum()
            x_bounds = (0, (n_epochs - len(rec.nb_batches)) * rec.nb_batches[-1] + len(rec.losses))
            y_bounds = (0, max((max(Tensor(rec.losses)), max(Tensor(rec.val_losses)))))
            rec.pbar.update_graph([(iters, rec.losses), (val_iter, rec.val_losses)], x_bounds, y_bounds)
            return False

class BnFreeze(LearnerCallback):
    "Freeze moving average statistics in all non-trainable batchnorm layers."
    def on_epoch_begin(self, **kwargs:Any)->None:
        "Put bn layers in eval mode on epoch_begin"
        set_bn_eval(self.learn.model)

@dataclass
class GradientClipping(LearnerCallback):
    "To do gradient clipping during training."
    clip:float

    def on_backward_end(self, **kwargs):
        if self.clip:  nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

