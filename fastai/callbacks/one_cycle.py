"Supports 1-Cycle style training"
from ..core import *
from ..callback import *
from ..basic_train import Learner,LearnerCallback

__all__ = ['OneCycleScheduler']

class OneCycleScheduler(LearnerCallback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."
    def __init__(self, learn:Learner, lr_max:float, moms:Floats=(0.95,0.85), div_factor:float=25., pct_start:float=0.3, 
                 tot_epochs:int=None, start_epoch:int=1):
        super().__init__(learn)
        self.lr_max,self.div_factor,self.pct_start = lr_max,div_factor,pct_start
        self.moms=tuple(listify(moms,2))
        if is_listy(self.lr_max): self.lr_max = np.array(self.lr_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs

    def steps(self, *steps_cfg:StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Stepper(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs + self.start_epoch - 1)
        n = len(self.learn.data.train_dl) * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n-a1
        self.phases = ((a1, annealing_cos), (a2, annealing_cos))
        low_lr = self.lr_max/self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, low_lr/1e4))
        self.mom_scheds = self.steps(self.moms, (self.moms[1], self.moms[0]))
        self.opt = self.learn.opt
        self.opt.lr,self.opt.mom = self.lr_scheds[0].start,self.mom_scheds[0].start
        self.idx_s = 0
        for _ in range(len(self.learn.data.train_dl) * (self.start_epoch-1)):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:
            if self.idx_s >= len(self.lr_scheds): return True
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Tell Learner to stop if the cycle is finished."
        return epoch + self.start_epoch - 1 > self.tot_epochs
