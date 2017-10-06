from .imports import *
from .layer_optimizer import *

class Callback:
    def on_train_begin(self, opt): pass
    def on_epoch_end(self, metrics): pass
    def on_batch_end(self, metrics): pass

class LossRecorder(Callback):
    def __init__(self, layer_opt):
        self.layer_opt=layer_opt
        self.init_lrs=np.array(layer_opt.lrs)
        self.on_train_begin()

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
        plt.plot(self.iterations[10:], self.losses[10:])

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lrs)


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
    def __init__(self, layer_opt, nb, end_lr=10):
        self.lr_mult = (end_lr/layer_opt.lr)**(1/nb)
        super().__init__(layer_opt)

    def on_train_begin(self):
        super().on_train_begin()
        self.best=1e9

    def calc_lr(self, init_lrs): return init_lrs * (self.lr_mult**self.iteration)

    def on_batch_end(self, loss):
        if math.isnan(loss) or loss>self.best*4:
            return True
        if loss<self.best: self.best=loss
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
            if self.on_cycle_end:
                self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs / 2 * cos_out
