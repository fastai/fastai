from .imports import *
from .torch_imports import *

def accuracy(preds, targs):
    preds = np.argmax(preds, axis=1)
    return (preds==targs).mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

