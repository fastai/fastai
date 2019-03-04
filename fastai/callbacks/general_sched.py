from ..core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

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
class GeneralScheduler(LearnerCallback):
    "Schedule multiple `TrainingPhase` for a `Learner`."
    def __init__(self, learn:Learner, phases:Collection[TrainingPhase], start_epoch:int=None):
        super().__init__(learn)
        self.phases,self.start_epoch = phases,start_epoch

    def on_train_begin(self, epoch:int, **kwargs:Any)->None:
        "Initialize the lr and mom schedules for training."
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.lr_scheds = [p.lr_step for p in self.phases]
        self.mom_scheds = [p.mom_step for p in self.phases]
        self.opt = self.learn.opt
        self.opt.lr,self.opt.mom = self.lr_scheds[0].start,self.mom_scheds[0].start
        self.idx_s = 0
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take a step in lr,mom sched, start next stepper when the current one is complete."
        if train:
            if self.idx_s >= len(self.lr_scheds): return {'stop_training': True, 'stop_epoch': True}
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1