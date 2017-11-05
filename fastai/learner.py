from .imports import *
from .torch_imports import *
from .core import *
from .transforms import *
from .model import *
from .dataset import *
from .sgdr import *
from .layer_optimizer import *
from .layers import *
from .metrics import *
from .losses import *
import time


class BasicModel():
    def __init__(self,model): self.model=model
    def get_layer_groups(self): return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self): return [self.model]


class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None):
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = os.path.join(self.data.path, tmp_name)
        self.models_path = os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit,self.reg_fn,self.crit = None,None,None

    #def num_features(self): return num_features(self.model)

    def __getitem__(self,i): return self.children[i]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): return model_summary(self.model, [3,self.data.sz,self.data.sz])

    def freeze_to(self, n):
        c=self.children
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def unfreeze(self): self.freeze_to(0)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    def save(self, name): save_model(self.model, self.get_model_path(name))
    def load(self, name): load_model(self.model, self.get_model_path(name))

    def set_data(self, data): self.data_ = data

    def get_cycle_end(self, name):
        if name is None: return None
        return lambda sched, cycle: self.save_cycle(name, cycle)

    def save_cycle(self, name, cycle): self.save(f'{name}_cyc_{cycle}')
    def load_cycle(self, name, cycle): self.load(f'{name}_cyc_{cycle}')

    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None,
                metrics=None, callbacks=None, **kwargs):
        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics
        if cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl)*cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched: self.sched=LossRecorder(layer_opt)
        callbacks+=[self.sched]
        for cb in callbacks: cb.on_train_begin()
        n_epoch = sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle)
        fit(model, data, n_epoch, layer_opt.opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, **kwargs)

    def get_layer_groups(self): return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None):
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr)
        self.fit_gen(self.model, self.data, layer_opt, 1)
        self.load('tmp')

    def predict(self, is_test=False): return self.predict_with_targs(is_test)[0]

    def predict_with_targs(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        return predict_with_targs(self.model, dl)

    def TTA(self, n_aug=4, is_test=False):
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1]*math.ceil(n_aug/4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in range(n_aug)]
        return np.stack(preds1+preds2).mean(0), targs

