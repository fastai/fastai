""" Helper functions for inspecting hyperparameters """

from fastai.basics import *

def get_lrsmoms_paired(learn):
    "Returns list of momentum at the end of each training epoch."
    ##Enp_learning_rates = [for lr in range(1, len(learn.recorder.lrs)] ##for i in range(1, len(list1)):
    batch_size = len(learn.data.train_dl)
    lrs = [batch[-1] for batch in partition(learn.recorder.lrs, batch_size)]
    moms = [batch[-1] for batch in partition(learn.recorder.moms, batch_size)]
    return zip(lrs,moms)

def get_momentum(learn):
    "Returns list of momentum at the end of each training epoch."
    ##Enp_learning_rates = [for lr in range(1, len(learn.recorder.lrs)] ##for i in range(1, len(list1)):
    batch_size = len(learn.data.train_dl)
    return [batch[-1] for batch in partition(learn.recorder.moms, batch_size)]

def get_learning_rates(learn):
    "Returns list of learning rates at the end of each training epoch."
    ##Enp_learning_rates = [for lr in range(1, len(learn.recorder.lrs)] ##for i in range(1, len(list1)):
    batch_size = len(learn.data.train_dl)
    return [batch[-1] for batch in partition(learn.recorder.lrs, batch_size)]
