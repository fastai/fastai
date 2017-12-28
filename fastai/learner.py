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
    def __init__(self,model,name='unnamed'): self.model,self.name = model,name
    def get_layer_groups(self, do_fc=False): return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self): return [self.model]


class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None):
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = os.path.join(self.data.path, tmp_name)
        self.models_path = os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit,self.reg_fn,self.crit = None,None,None

    @classmethod
    def from_model_data(cls, m, data):
        self = cls(data, BasicModel(to_gpu(m)))
        self.unfreeze()
        return self

    def __getitem__(self,i): return self.children[i]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): return model_summary(self.model, [3,self.data.sz,self.data.sz])

    def __repr__(self): return self.model.__repr__()

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'): m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c=self.get_layer_groups()
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
                metrics=None, callbacks=None, use_wd_sched=False, norm_wds=False, wds_sched_mult=None, **kwargs):
        """Method does some preparation before finally delegating to the 'fit' method for
        fitting the model. Namely, if cycle_len is defined, it adds a 'Cosine Annealing'
        scheduler for varying the learning rate across iterations.

        Method also computes the total number of epochs to fit based on provided 'cycle_len',
        'cycle_mult', and 'n_cycle' parameters.

        Args:
            model (Learner):  Any neural architecture for solving a supported problem.
                Eg. ResNet-34, RNN_Learner etc.

            data (ModelData): An instance of ModelData.

            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class

            n_cycle (int): number of cycles

            cycle_len (int):  number of cycles before lr is reset to the initial value.
                E.g if cycle_len = 3, then the lr is varied between a maximum
                and minimum value over 3 epochs.

            cycle_mult (int): additional parameter for influencing how the lr resets over
                the cycles. For an intuitive explanation, please see
                https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb

            cycle_save_name (str): use to save the weights at end of each cycle

            metrics (function): some function for evaluating a desired metric. Eg. accuracy.

            callbacks (list(Callback)): callbacks to apply during the training.

            use_wd_sched (bool, optional): set to True to enable weight regularization using
                the technique mentioned in https://arxiv.org/abs/1711.05101. When this is True
                alone (see below), the regularization is detached from gradient update and
                applied directly to the weights.

            norm_wds (bool, optional): when this is set to True along with use_wd_sched, the
                regularization factor is normalized with each training cycle.

            wds_sched_mult (function, optional): when this is provided along with use_wd_sched
                as True, the value computed by this function is multiplied with the regularization
                strength. This function is passed the WeightDecaySchedule object. And example
                function that can be passed is:
                            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs

            kwargs: other optional arguments

        Returns:
            None
        """

        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics

        if use_wd_sched:
            # This needs to come before CosAnneal() because we need to read the initial learning rate from
            # layer_opt.lrs - but CosAnneal() alters the layer_opt.lrs value initially (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print('fit() warning: use_wd_sched is set to True, but weight decay(s) passed are 0. Use wds to '
                      'pass weight decay values.')
            batch_per_epoch = len(data.trn_dl)
            cl = cycle_len if cycle_len else 1
            self.wd_sched = WeightDecaySchedule(layer_opt, batch_per_epoch, cl, cycle_mult, n_cycle,
                                                norm_wds, wds_sched_mult)
            callbacks += [self.wd_sched]

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

        """Method returns an instance of the LayerOptimizer class, which
        allows for setting differential learning rates for different
        parts of the model.

        An example of how a model maybe differentiated into different parts
        for application of differential learning rates and weight decays is
        seen in ../.../courses/dl1/fastai/conv_learner.py, using the dict
        'model_meta'. Currently, this seems supported only for convolutional
        networks such as VGG-19, ResNet-XX etc.

        Args:
            lrs (float or list(float)): learning rate(s) for the model

            wds (float or list(float)): weight decay parameter(s).

        Returns:
            An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):

        """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

        Note that one can specify a list of learning rates which, when appropriately
        defined, will be applied to different segments of an architecture. This seems
        mostly relevant to ImageNet-trained models, where we want to alter the layers
        closest to the images by much smaller amounts.

        Likewise, a single or list of weight decay parameters can be specified, which
        if appropriate for a model, will apply variable weight decay parameters to
        different segments of the model.

        Args:
            lrs (float or list(float)): learning rate for the model

            n_cycle (int): number of cycles (or iterations) to fit the model for

            wds (float or list(float)): weight decay parameter(s).

            kwargs: other arguments

        Returns:
            None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def warm_up(self, start_lr=1e-5, end_lr=10, wds=None):
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr)
        self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None):
        """Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:

            >> learn.lr_find(lr/1000)

            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.

        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186

        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr)
        self.fit_gen(self.model, self.data, layer_opt, 1)
        self.load('tmp')

    def predict(self, is_test=False): return self.predict_with_targs(is_test)[0]

    def predict_with_targs(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        return predict_with_targs(self.model, dl)

    def predict_dl(self, dl): return predict_with_targs(self.model, dl)[0]
    def predict_array(self, arr): return to_np(self.model(V(T(arr).cuda())))

    def TTA(self, n_aug=4, is_test=False):
        """ Predict with Test Time Augmentation (TTA)

        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        Args:
            n_aug: a number of augmentation images to use per original image
            is_test: indicate to use test images; otherwise use validation images

        Returns:
            (tuple): a tuple containing:

                log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
        """
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1]*math.ceil(n_aug/4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in tqdm(range(n_aug), leave=False)]
        return np.stack(preds1+preds2), targs

