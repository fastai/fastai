"Supports flat-cosine-annealling style training"

from ..core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['FlatCosAnnealScheduler']

# The brain child of Mikhail Grankin aimed for use of the new optimizers
# For more information: https://forums.fast.ai/t/how-we-beat-the-5-epoch-imagewoof-leaderboard-score-some-new-techniques-to-consider/53453

def FlatCosAnnealScheduler(learn, lr:float=4e-3, tot_epochs:int=1, moms:Floats=(0.95,0.999),
                          start_pct:float=0.72, curve='cosine'):
  "Manage FCFit trainnig as found in the ImageNette experiments"
  n = len(learn.data.train_dl)
  anneal_start = int(n * tot_epochs * start_pct)
  batch_finish = ((n * tot_epochs) - anneal_start)
  if curve=="cosine":
      curve_type=annealing_cos
  elif curve=="linear":
      curve_type=annealing_linear
  elif curve=="exponential":
      curve_type=annealing_exp
  else:
      raiseValueError(f"annealing type not supported {curve}")
      
  phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr).schedule_hp('mom', moms[0])
  phase1 = TrainingPhase(batch_finish).schedule_hp('lr', lr, anneal=curve_type).schedule_hp('mom', moms[1])
  phases = [phase0, phase1]
  return GeneralScheduler(learn, phases)
