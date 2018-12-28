from ..core import *
from ..callback import *
from ..basic_train import Learner

__all__ = ['GeneralScheduler', 'TrainingPhase']

@dataclass
class TrainingPhase():
    "Schedule `lrs` and `moms` according to `lr_anneal` and `mom_anneal` across a `length` schedule."
    length:int
    lrs:Floats
    moms:Floats
    lr_anneal:AnnealFunc=None
    mom_anneal:AnnealFunc=None

    def __post_init__(self)->None:
        self.lr_step = Stepper(self.lrs, self.length, self.lr_anneal)
        self.mom_step = Stepper(self.moms, self.length, self.mom_anneal)

@dataclass
class GeneralScheduler(Callback):
    "Schedule multiple `TrainingPhase` for a `Learner`."
    learn:Learner
    phases:Collection[TrainingPhase]

    def on_train_begin(self, n_epochs:int, **kwargs:Any)->None:
        "Initialize the lr and mom schedules for training."
        self.lr_scheds = [p.lr_step for p in self.phases]
        self.mom_scheds = [p.mom_step for p in self.phases]
        self.opt = self.learn.opt
        self.opt.lr,self.opt.mom = self.lr_scheds[0].start,self.mom_scheds[0].start
        self.idx_s = 0

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take a step in lr,mom sched, start next stepper when the current one is complete."
        if train:
            if self.idx_s >= len(self.lr_scheds): return True
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1