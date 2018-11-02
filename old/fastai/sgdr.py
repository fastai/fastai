from .imports import *
from .layer_optimizer import *
from enum import IntEnum
from timeit import default_timer as timer
import copy
import math


class Callback:
    '''
    An abstract class that all callback(e.g., LossRecorder) classes extends from. 
    Must be extended before usage.
    '''
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics): pass
    def on_phase_end(self): pass
    def on_batch_end(self, metrics): pass
    def on_train_end(self): pass

# Useful for maintaining status of a long-running job.
# 
# Usage:
# learn.fit(0.01, 1, callbacks = [LoggingCallback(save_path="/tmp/log")])
class LoggingCallback(Callback):
    '''
    A class useful for maintaining status of a long-running job.
    e.g.: learn.fit(0.01, 1, callbacks = [LoggingCallback(save_path="/tmp/log")])
    '''
    def __init__(self, save_path):
        super().__init__()
        self.save_path=save_path
    def on_train_begin(self):
        self.batch = 0
        self.epoch = 0
        self.phase = 0
        self.f = open(self.save_path, "a", 1)
        self.log("\ton_train_begin")
    def on_batch_begin(self):
        self.log(str(self.batch)+"\ton_batch_begin")
    def on_phase_begin(self):
        self.log(str(self.phase)+"\ton_phase_begin")
    def on_epoch_end(self, metrics):
        self.log(str(self.epoch)+"\ton_epoch_end: "+str(metrics))
        self.epoch += 1
    def on_phase_end(self):
        self.log(str(self.phase)+"\ton_phase_end")
        self.phase+=1
    def on_batch_end(self, metrics):
        self.log(str(self.batch)+"\ton_batch_end: "+str(metrics))
        self.batch += 1
    def on_train_end(self):
        self.log("\ton_train_end")
        self.f.close()
    def log(self, string):
        self.f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+"\t"+string+"\n")
        
class LossRecorder(Callback):
    '''
    Saves and displays loss functions and other metrics. 
    Default sched when none is specified in a learner. 
    '''
    def __init__(self, layer_opt, save_path='', record_mom=False, metrics=[]):
        super().__init__()
        self.layer_opt=layer_opt
        self.init_lrs=np.array(layer_opt.lrs)
        self.save_path, self.record_mom, self.metrics = save_path, record_mom, metrics

    def on_train_begin(self):
        self.losses,self.lrs,self.iterations,self.epochs,self.times = [],[],[],[],[]
        self.start_at = timer()
        self.val_losses, self.rec_metrics = [], []
        if self.record_mom:
            self.momentums = []
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1
        self.epochs.append(self.iteration)
        self.times.append(timer() - self.start_at)
        self.save_metrics(metrics)

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iterations.append(self.iteration)
        if isinstance(loss, list):
            self.losses.append(loss[0])
            self.save_metrics(loss[1:])
        else: self.losses.append(loss)
        if self.record_mom: self.momentums.append(self.layer_opt.mom)

    def save_metrics(self,vals):
        self.val_losses.append(delistify(vals[0]))
        if len(vals) > 2: self.rec_metrics.append(vals[1:])
        elif len(vals) == 2: self.rec_metrics.append(vals[1])

    def plot_loss(self, n_skip=10, n_skip_end=5):
        '''
        plots loss function as function of iterations. 
        When used in Jupyternotebook, plot will be displayed in notebook. Else, plot will be displayed in console and both plot and loss are saved in save_path. 
        '''
        if not in_ipynb(): plt.switch_backend('agg')
        plt.plot(self.iterations[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
            np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_lr(self):
        '''Plots learning rate in jupyter notebook or console, depending on the enviroment of the learner.'''
        if not in_ipynb():
            plt.switch_backend('agg')
        if self.record_mom:
            fig, axs = plt.subplots(1,2,figsize=(12,4))
            for i in range(0,2): axs[i].set_xlabel('iterations')
            axs[0].set_ylabel('learning rate')
            axs[1].set_ylabel('momentum')
            axs[0].plot(self.iterations,self.lrs)
            axs[1].plot(self.iterations,self.momentums)   
        else:
            plt.xlabel("iterations")
            plt.ylabel("learning rate")
            plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


class LR_Updater(LossRecorder):
    '''
    Abstract class where all Learning Rate updaters inherit from. (e.g., CirularLR)
    Calculates and updates new learning rate and momentum at the end of each batch. 
    Have to be extended. 
    '''
    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()
        if self.record_mom:
            self.update_mom()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        if self.record_mom:
            self.update_mom()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)
    
    def update_mom(self):
        new_mom = self.calc_mom()
        self.layer_opt.set_mom(new_mom)

    @abstractmethod
    def calc_lr(self, init_lrs): raise NotImplementedError
    
    @abstractmethod
    def calc_mom(self): raise NotImplementedError


class LR_Finder(LR_Updater):
    '''
    Helps you find an optimal learning rate for a model, as per suggetion of 2015 CLR paper. 
    Learning rate is increased in linear or log scale, depending on user input, and the result of the loss funciton is retained and can be plotted later. 
    '''
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics = []):
        self.linear, self.stop_dv = linear, True
        ratio = end_lr/layer_opt.lr
        self.lr_mult = (ratio/nb) if linear else ratio**(1/nb)
        super().__init__(layer_opt,metrics=metrics)

    def on_train_begin(self):
        super().on_train_begin()
        self.best=1e9

    def calc_lr(self, init_lrs):
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration
        return init_lrs * mult

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics,list) else metrics
        if self.stop_dv and (math.isnan(loss) or loss>self.best*4):
            return True
        if (loss<self.best and self.iteration>10): self.best=loss
        return super().on_batch_end(metrics)

    def plot(self, n_skip=10, n_skip_end=5):
        '''
        Plots the loss function with respect to learning rate, in log scale. 
        '''
        plt.ylabel("validation loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-(n_skip_end+1)], self.losses[n_skip:-(n_skip_end+1)])
        plt.xscale('log')

class LR_Finder2(LR_Finder):
    """
        A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
    """
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics=[], stop_dv=True):
        self.nb, self.metrics = nb, metrics
        super().__init__(layer_opt, nb, end_lr, linear, metrics)
        self.stop_dv = stop_dv

    def on_batch_end(self, loss):
        if self.iteration == self.nb:
            return True
        return super().on_batch_end(loss)

    def plot(self, n_skip=10, n_skip_end=5, smoothed=True):
        if self.metrics is None: self.metrics = []
        n_plots = len(self.metrics)+2
        fig, axs = plt.subplots(n_plots,figsize=(6,4*n_plots))
        for i in range(0,n_plots): axs[i].set_xlabel('learning rate')
        axs[0].set_ylabel('training loss')
        axs[1].set_ylabel('validation loss')
        for i,m in enumerate(self.metrics): 
            axs[i+2].set_ylabel(m.__name__)
            if len(self.metrics) == 1:
                values = self.rec_metrics
            else:
                values = [rec[i] for rec in self.rec_metrics]
            if smoothed: values = smooth_curve(values,0.98)
            axs[i+2].plot(self.lrs[n_skip:-n_skip_end], values[n_skip:-n_skip_end])
        plt_val_l = smooth_curve(self.val_losses, 0.98) if smoothed else self.val_losses
        axs[0].plot(self.lrs[n_skip:-n_skip_end],self.losses[n_skip:-n_skip_end])
        axs[1].plot(self.lrs[n_skip:-n_skip_end],plt_val_l[n_skip:-n_skip_end])

class CosAnneal(LR_Updater):
    ''' Learning rate scheduler that implements a cosine annealation schedule. '''
    def __init__(self, layer_opt, nb, on_cycle_end=None, cycle_mult=1):
        self.nb,self.on_cycle_end,self.cycle_mult = nb,on_cycle_end,cycle_mult
        super().__init__(layer_opt)

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.iteration<self.nb/20:
            self.cycle_iter += 1
            return init_lrs/100.

        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs / 2 * cos_out


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme. 
    Learning rate is increased then decreased linearly. 
    '''
    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None, momentums=None):
        self.nb,self.div,self.cut_div,self.on_cycle_end = nb,div,cut_div,on_cycle_end
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt)/(self.nb - cut_pt)
        else: pct = self.cycle_iter/cut_pt
        res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res
    
    def calc_mom(self):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = (self.cycle_iter - cut_pt)/(self.nb - cut_pt)
        else: pct = 1 - self.cycle_iter/cut_pt
        res = self.moms[1] + pct * (self.moms[0] - self.moms[1])
        return res

class CircularLR_beta(LR_Updater):
    def __init__(self, layer_opt, nb, div=10, pct=10, on_cycle_end=None, momentums=None):
        self.nb,self.div,self.pct,self.on_cycle_end = nb,div,pct,on_cycle_end
        self.cycle_nb = int(nb * (1-pct/100) / 2)
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.cycle_iter>2 * self.cycle_nb:
            pct = (self.cycle_iter - 2*self.cycle_nb)/(self.nb - 2*self.cycle_nb)
            res = init_lrs * (1 + (pct * (1-100)/100)) / self.div
        elif self.cycle_iter>self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb)/self.cycle_nb
            res = init_lrs * (1 + pct*(self.div-1)) / self.div
        else:
            pct = self.cycle_iter/self.cycle_nb
            res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_mom(self):
        if self.cycle_iter>2*self.cycle_nb:
            res = self.moms[0]
        elif self.cycle_iter>self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb)/self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        else:
            pct = self.cycle_iter/self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        return res


class SaveBestModel(LossRecorder):
    
    """ Save weights of the best model based during training.
        If metrics are provided, the first metric in the list is used to
        find the best model. 
        If no metrics are provided, the loss is used.
        
        Args:
            model: the fastai model
            lr: indicate to use test images; otherwise use validation images
            name: the name of filename of the weights without '.h5'
        
        Usage:
            Briefly, you have your model 'learn' variable and call fit.
            >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='mybestmodel')
            ....
            >>> learn.load('mybestmodel')
            
            For more details see http://forums.fast.ai/t/a-code-snippet-to-save-the-best-model-during-training/12066
 
    """
    def __init__(self, model, layer_opt, metrics, name='best_model'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_only_loss if metrics==None else self.save_when_acc
        
    def save_when_only_loss(self, metrics):
        loss = metrics[0]
        if self.best_loss == None or loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
    
    def save_when_acc(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
        
    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        if math.isnan(metrics[0]): return
        self.save_method(metrics)


class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, batch_per_epoch, cycle_len, cycle_mult, n_cycles, norm_wds=False, wds_sched_mult=None):
        """
        Implements the weight decay schedule as mentioned in https://arxiv.org/abs/1711.05101

        :param layer_opt: The LayerOptimizer
        :param batch_per_epoch: Num batches in 1 epoch
        :param cycle_len: Num epochs in initial cycle. Subsequent cycle_len = previous cycle_len * cycle_mult
        :param cycle_mult: Cycle multiplier
        :param n_cycles: Number of cycles to be executed
        """
        super().__init__()

        self.layer_opt = layer_opt
        self.batch_per_epoch = batch_per_epoch
        self.init_wds = np.array(layer_opt.wds)  # Weights as set by user
        self.init_lrs = np.array(layer_opt.lrs)  # Learning rates as set by user
        self.new_wds = None                      # Holds the new weight decay factors, calculated in on_batch_begin()
        self.iteration = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # Pre calculating the number of epochs in the cycle of current running epoch
        self.epoch_to_num_cycles, i = dict(), 0
        for cycle in range(n_cycles):
            for _ in range(cycle_len):
                self.epoch_to_num_cycles[i] = cycle_len
                i += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0

    def on_batch_begin(self):
        # Prepare for decay of weights

        # Default weight decay (as provided by user)
        wdn = self.init_wds

        # Weight decay multiplier (The 'eta' in the paper). Optional.
        wdm = 1.0
        if self.wds_sched_mult is not None:
            wdm = self.wds_sched_mult(self)

        # Weight decay normalized. Optional.
        if self.norm_wds:
            wdn = wdn / np.sqrt(self.batch_per_epoch * self.epoch_to_num_cycles[self.epoch])

        # Final wds
        self.new_wds = wdm * wdn

        # Set weight_decay with zeros so that it is not applied in Adam, we will apply it outside in on_batch_end()
        self.layer_opt.set_wds_out(self.new_wds)
        # We have to save the existing weights before the optimizer changes the values
        self.iteration += 1

    def on_epoch_end(self, metrics):
        self.epoch += 1

class DecayType(IntEnum):
    ''' Data class, each decay type is assigned a number. '''
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5

class DecayScheduler():
    '''Given initial and endvalue, this class generates the next value depending on decay type and number of iterations. (by calling next_val().) '''

    def __init__(self, dec_type, num_it, start_val, end_val=None, extra=None):
        self.dec_type, self.nb, self.start_val, self.end_val, self.extra = dec_type, num_it, start_val, end_val, extra
        self.it = 0
        if self.end_val is None and not (self.dec_type in [1,4]): self.end_val = 0
    
    def next_val(self):
        self.it += 1
        if self.dec_type == DecayType.NO:
            return self.start_val
        elif self.dec_type == DecayType.LINEAR:
            pct = self.it/self.nb
            return self.start_val + pct * (self.end_val-self.start_val)
        elif self.dec_type == DecayType.COSINE:
            cos_out = np.cos(np.pi*(self.it)/self.nb) + 1
            return self.end_val + (self.start_val-self.end_val) / 2 * cos_out
        elif self.dec_type == DecayType.EXPONENTIAL:
            ratio = self.end_val / self.start_val
            return self.start_val * (ratio **  (self.it/self.nb))
        elif self.dec_type == DecayType.POLYNOMIAL:
            return self.end_val + (self.start_val-self.end_val) * (1 - self.it/self.nb)**self.extra
        

class TrainingPhase():
    '''
    Object with training information for each phase, when multiple phases are involved during training.  
    Used in fit_opt_sched in learner.py
    '''
    def __init__(self, epochs=1, opt_fn=optim.SGD, lr=1e-2, lr_decay=DecayType.NO, momentum=0.9,
                momentum_decay=DecayType.NO, beta=None, wds=None, wd_loss=True):
        """
        Creates an object containing all the relevant informations for one part of a model training.

        Args
        epochs: number of epochs to train like this
        opt_fn: an optimizer (example optim.Adam)
        lr: one learning rate or a tuple of the form (start_lr,end_lr)
          each of those can be a list/numpy array for differential learning rates
        lr_decay: a DecayType object specifying how the learning rate should change
        momentum: one momentum (or beta1 in case of Adam), or a tuple of the form (start_mom,end_mom)
        momentum_decay: a DecayType object specifying how the momentum should change
        beta: beta2 parameter of Adam or alpha parameter of RMSProp
        wds: weight decay (can be an array for differential wds)
        """
        self.epochs, self.opt_fn, self.lr, self.momentum, self.beta, self.wds = epochs, opt_fn, lr, momentum, beta, wds
        if isinstance(lr_decay,tuple): self.lr_decay, self.extra_lr = lr_decay
        else: self.lr_decay, self.extra_lr = lr_decay, None
        if isinstance(momentum_decay,tuple): self.mom_decay, self.extra_mom = momentum_decay
        else: self.mom_decay, self.extra_mom = momentum_decay, None
        self.wd_loss = wd_loss

    def phase_begin(self, layer_opt, nb_batches):
        self.layer_opt = layer_opt
        if isinstance(self.lr, tuple): start_lr,end_lr = self.lr
        else: start_lr, end_lr = self.lr, None
        self.lr_sched = DecayScheduler(self.lr_decay, nb_batches * self.epochs, start_lr, end_lr, extra=self.extra_lr)
        if isinstance(self.momentum, tuple): start_mom,end_mom = self.momentum
        else: start_mom, end_mom = self.momentum, None
        self.mom_sched = DecayScheduler(self.mom_decay, nb_batches * self.epochs, start_mom, end_mom, extra=self.extra_mom)
        self.layer_opt.set_opt_fn(self.opt_fn)
        self.layer_opt.set_lrs(start_lr)
        self.layer_opt.set_mom(start_mom)
        if self.beta is not None: self.layer_opt.set_beta(self.beta)
        if self.wds is not None:
            if self.wd_loss: self.layer_opt.set_wds(self.wds)
            else: self.layer_opt.set_wds_out(self.wds)
    
    def update(self):
        new_lr, new_mom = self.lr_sched.next_val(), self.mom_sched.next_val()
        self.layer_opt.set_lrs(new_lr)
        self.layer_opt.set_mom(new_mom)
    

class OptimScheduler(LossRecorder):
    '''Learning rate Scheduler for training involving multiple phases.'''

    def __init__(self, layer_opt, phases, nb_batches, stop_div = False):
        self.phases, self.nb_batches, self.stop_div = phases, nb_batches, stop_div
        super().__init__(layer_opt, record_mom=True)

    def on_train_begin(self):
        super().on_train_begin()
        self.phase,self.best=0,1e9

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics,list) else metrics
        if self.stop_div and (math.isnan(loss) or loss>self.best*4):
            return True
        if (loss<self.best and self.iteration>10): self.best=loss
        super().on_batch_end(metrics)
        self.phases[self.phase].update()
    
    def on_phase_begin(self):
        self.phases[self.phase].phase_begin(self.layer_opt, self.nb_batches[self.phase])

    def on_phase_end(self):
        self.phase += 1

    def plot_lr(self, show_text=True, show_moms=True):
        """Plots the lr rate/momentum schedule"""
        phase_limits = [0]
        for nb_batch, phase in zip(self.nb_batches, self.phases):
            phase_limits.append(phase_limits[-1] + nb_batch * phase.epochs)
        if not in_ipynb():
            plt.switch_backend('agg')
        np_plts = 2 if show_moms else 1
        fig, axs = plt.subplots(1,np_plts,figsize=(6*np_plts,4))
        if not show_moms: axs = [axs]
        for i in range(np_plts): axs[i].set_xlabel('iterations')
        axs[0].set_ylabel('learning rate')
        axs[0].plot(self.iterations,self.lrs)
        if show_moms:
            axs[1].set_ylabel('momentum')
            axs[1].plot(self.iterations,self.momentums)
        if show_text:   
            for i, phase in enumerate(self.phases):
                text = phase.opt_fn.__name__
                if phase.wds is not None: text+='\nwds='+str(phase.wds)
                if phase.beta is not None: text+='\nbeta='+str(phase.beta)
                for k in range(np_plts):
                    if i < len(self.phases)-1:
                        draw_line(axs[k], phase_limits[i+1])
                    draw_text(axs[k], (phase_limits[i]+phase_limits[i+1])/2, text) 
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))
    
    def plot(self, n_skip=10, n_skip_end=5, linear=None):
        if linear is None: linear = self.phases[-1].lr_decay == DecayType.LINEAR
        plt.ylabel("loss")
        plt.plot(self.lrs[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if linear: plt.xlabel("learning rate")
        else:
            plt.xlabel("learning rate (log scale)")
            plt.xscale('log')

def draw_line(ax,x):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot([x,x],[ymin,ymax], color='red', linestyle='dashed')

def draw_text(ax,x, text):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(x,(ymin+ymax)/2,text, horizontalalignment='center', verticalalignment='center', fontsize=14, alpha=0.5)

def smooth_curve(vals, beta):
    avg_val = 0
    smoothed = []
    for (i,v) in enumerate(vals):
        avg_val = beta * avg_val + (1-beta) * v
        smoothed.append(avg_val/(1-beta**(i+1)))
    return smoothed
