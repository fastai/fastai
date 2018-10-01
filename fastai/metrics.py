"Implements various metrics to measure training accuracy"
from .torch_core import *

__all__ = ['accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'fbeta']

def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, beta:float=2, eps:float=1e-9, sigmoid:bool=True) -> Rank0Tensor:
    "Compute the f_beta between preds and targets."
    beta2 = beta**2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True) -> Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean()

def dice(input:Tensor, targs:Tensor) -> Rank0Tensor:
    "Dice coefficient metric for binary target."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    return 2. * intersect / union

def accuracy(input:Tensor, targs:Tensor) -> Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def exp_rmspe(pred:Tensor, targ:Tensor) -> Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())
