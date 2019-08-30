"Supports flat-cosine-annealling style training"

from ..core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['FlatCosAnnealScheduler']

# The brain child of Mikhail Grankin aimed for use of the new optimizers
# For more information: https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider/53453

class FlatCosAnnealScheduler(LearnerCallback):
    """
    Manage FCFit training as found in the ImageNette experiments. 
    https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider/53453
    Code format is based on OneCycleScheduler
    Based on idea by Mikhail Grankin
    """
    def __init__(self, learn:Learner, lr:float=4e-3, moms:Floats=(0.95,0.999),
               start_pct:float=0.72, start_epoch:int=None, tot_epochs:int=None,
                curve='cosine'):
        super().__init__(learn)
        n = len(learn.data.train_dl)
        self.anneal_start = int(n * tot_epochs * start_pct)
        self.batch_finish = (n * tot_epochs - self.anneal_start)
        if curve=="cosine":
            curve_type=annealing_cos
        elif curve=="linear":
            curve_type=annealing_linear
        elif curve=="exponential":
            curve_type=annealing_exp
        else:
            raiseValueError(f"annealing type not supported {curve}")
        phase0 = TrainingPhase(self.anneal_start).schedule_hp('lr', lr).schedule_hp('mom', moms[0])
        phase1 = TrainingPhase(self.batch_finish).schedule_hp('lr', lr, anneal=curve_type).schedule_hp('mom', moms[1])
        phases = [phase0, phase1]
        self.phases,self.start_epoch = phases,start_epoch

        
    def on_train_begin(self, epoch:int, **kwargs:Any)->None:
        "Initialize the schedulers for training."
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.scheds = [p.scheds for p in self.phases]
        self.opt = self.learn.opt
        for k,v in self.scheds[0].items(): 
            v.restart()
            self.opt.set_stat(k, v.start)
        self.idx_s = 0
        return res
    
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

            
    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take a step in lr,mom sched, start next stepper when the current one is complete."
        if train:
            if self.idx_s >= len(self.scheds): return {'stop_training': True, 'stop_epoch': True}
            sched = self.scheds[self.idx_s]
            for k,v in sched.items(): self.opt.set_stat(k, v.step())
            if list(sched.values())[0].is_done: self.idx_s += 1