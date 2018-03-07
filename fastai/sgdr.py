from .imports import *
from .layer_optimizer import *
import copy


class Callback:
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_epoch_end(self, metrics): pass
    def on_batch_end(self, metrics): pass
    def on_train_end(self): pass
    
# Useful for maintaining status of a long-running job.
# 
# Usage:
# learn.fit(0.01, 1, callbacks = [LoggingCallback(save_path="/tmp/log")])
class LoggingCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path=save_path
    def on_train_begin(self): 
        self.batch = 0
        self.epoch = 0
        self.f = open(self.save_path, "a", 1)
        self.log("\ton_train_begin")
    def on_batch_begin(self): 
        self.log(str(self.batch)+"\ton_batch_begin")
    def on_epoch_end(self, metrics): 
        self.log(str(self.epoch)+"\ton_epoch_end: "+str(metrics))
        self.epoch += 1
    def on_batch_end(self, metrics): 
        self.log(str(self.batch)+"\ton_batch_end: "+str(metrics))
        self.batch += 1
    def on_train_end(self): 
        self.log("\ton_train_end")
        self.f.close()
    def log(self, string):
        self.f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+"\t"+string+"\n")

class LossRecorder(Callback):
    def __init__(self, layer_opt, save_path=''):
        super().__init__()
        self.layer_opt=layer_opt
        self.init_lrs=np.array(layer_opt.lrs)
        self.save_path=save_path

    def on_train_begin(self):
        self.losses,self.lrs,self.iterations = [],[],[]
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iterations.append(self.iteration)
        self.losses.append(loss)

    def plot_loss(self):
        if not in_ipynb():
            plt.switch_backend('agg')
        plt.plot(self.iterations[10:], self.losses[10:])
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
            np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_lr(self):
        if not in_ipynb():
            plt.switch_backend('agg')
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


class LR_Updater(LossRecorder):
    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)

    @abstractmethod
    def calc_lr(self, init_lrs): raise NotImplementedError


class LR_Finder(LR_Updater):
    def __init__(self, layer_opt, nb, end_lr=10, linear=False):
        self.linear = linear
        ratio = end_lr/layer_opt.lr
        self.lr_mult = (ratio/nb) if linear else ratio**(1/nb)
        super().__init__(layer_opt)

    def on_train_begin(self):
        super().on_train_begin()
        self.best=1e9

    def calc_lr(self, init_lrs):
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration
        return init_lrs * mult

    def on_batch_end(self, loss):
        if math.isnan(loss) or loss>self.best*4:
            return True
        if (loss<self.best and self.iteration>10): self.best=loss
        return super().on_batch_end(loss)

    def plot(self, n_skip=10):
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-5], self.losses[n_skip:-5])
        plt.xscale('log')


class CosAnneal(LR_Updater):
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
    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None):
        self.nb,self.div,self.cut_div,self.on_cycle_end = nb,div,cut_div,on_cycle_end
        super().__init__(layer_opt)

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt)/(cut_pt*(self.cut_div-1))
        else: pct = self.cycle_iter/cut_pt
        res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class SaveBestModel(LossRecorder):
    
    """ Save weigths of the model with
        the best accuracy during training.
        
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
    def __init__(self, model, layer_opt, name='best_model'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss, acc = metrics
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')


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
        self.param_groups_old = None             # Caches the old parameter values in on_batch_begin()
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

        # Record the wds
        self.wds_history.append(self.new_wds)

        # Set weight_decay with zeros so that it is not applied in Adam, we will apply it outside in on_batch_end()
        self.layer_opt.set_wds(torch.zeros(self.new_wds.size))
        # We have to save the existing weights before the optimizer changes the values
        self.param_groups_old = copy.deepcopy(self.layer_opt.opt.param_groups)
        self.iteration += 1

    def on_batch_end(self, loss):
        # Decay the weights
        for group, group_old, wds in zip(self.layer_opt.opt.param_groups, self.param_groups_old, self.new_wds):
            for p, p_old in zip(group['params'], group_old['params']):
                if p.grad is None:
                    continue
                p.data = p.data.add(-wds, p_old.data)

    def on_epoch_end(self, metrics):
        self.epoch += 1
