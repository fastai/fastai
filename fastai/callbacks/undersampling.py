from ..torch_core import *
from ..basic_data import DataBunch
from ..callback import *
from ..basic_train import Learner,LearnerCallback
from torch.utils.data.sampler import WeightedRandomSampler

__all__ = ['UnderSamplingCallback']

class UnderSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner,weights:torch.Tensor=None):
        super().__init__(learn)
        self.weights = weights

    def on_train_begin(self, **kwargs):
        self.old_dl = self.data.train_dl
        self.labels = self.data.train_dl.y.items
        assert np.issubdtype(self.labels.dtype, np.integer), "Can only undersample integer values"
        _,self.label_counts = np.unique(self.labels,return_counts=True)
        if self.weights is None: self.weights = torch.DoubleTensor((1/self.label_counts)[self.labels])
        self.total_len_undersample = int(self.data.c*np.min(self.label_counts))
        sampler = WeightedRandomSampler(self.weights, self.total_len_undersample)
        self.data.train_dl = self.data.train_dl.new(shuffle=False, sampler=sampler)
    
    def on_train_end(self, **kwargs):
        "Reset dataloader to its original state"
        self.data.train_dl = self.old_dl
