"Implements various metrics to measure training accuracy"
from .torch_core import *
from .callback import *

__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'accuracy_balanced', 'top_k_accuracy', 
           'mean_class_accuracy', 'dice', 'kappa_score', 'confusion_matrix', 'mae', 'mse', 
           'msle', 'rmse', 'explained_variance', 'r2_score', 'exp_rmspe', 'fbeta']


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()


def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()


def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean()


def accuracy_balanced(input:Tensor, targs:Tensor, clw:list=None)->Rank0Tensor:
    "Balanced accuracy score between `input` and `targs` w.r.t. class label weights `clw`."
    n = targs.shape[0]
    if not clw:
        clw = [1 for _ in range(input.shape[-1])]
    clw = Tensor(clw).view(1,-1).transpose(1, 0)
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    clw = clw / (torch.sum((input == targs).float() * clw))
    return torch.sum((input == targs).float() * clw) / clw.sum()


def top_k_accuracy(input:Tensor, targs:Tensor, k:int=5)->Rank0Tensor:
    "Computes the Top-k accuracy (target is in the top k predictions)."
    n = targs.shape[0]
    input = input.topk(k=k, dim=-1)[1].view(n, -1)
    targs = targs.view(n,-1)
    return (input == targs).sum(dim=1).float().mean()


def mean_class_accuracy(input: Tensor, targs: Tensor):
    "Computes the accuracy for each class label ->Rank1Tensor"
    x = torch.arange(0, input.shape[-1])
    targs = targs==x[:,None]
    input = input.argmax(-1)==x[:,None]
    label_sum = targs.sum(dim=1).float()
    eq = targs.float() * input.float()
    return eq.sum(1) / label_sum


def error_rate(input:Tensor, targs:Tensor)->Rank0Tensor:
    "1 - `accuracy`"
    return 1-accuracy(input, targs)


def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return 2. * intersect / union
    else: return intersect / (union-intersect+1.0)


def kappa_score(pred:Tensor, rater:Tensor)->Rank0Tensor:
    "Computes the rate of agreement (Cohens Kappa) between `pred` and `rater`."
    n = pred.shape[-1]
    c = confusion_matrix(pred, rater).float()
    sum0 = c.sum(0)
    sum1 = c.sum(1)
    expected = torch.einsum('i,j->ij', (sum0, sum1)) / torch.sum(sum0)
    w = torch.ones((n, n))
    idx = torch.arange(0, n)
    w[idx, idx] = 0
    k = torch.sum(w * c) / torch.sum(w * expected)
    return 1 - k


def confusion_matrix(input:Tensor, targs:Tensor):
    "Computes the confusion matrix."
    x = torch.arange(0, input.shape[-1])
    input = input.argmax(dim=-1).view(-1)
    cm = ((input==x[:, None]) & (targs==x[:, None, None])).sum(2)
    return cm


def exp_rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())


def mae(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean absolute error between `pred` and `targ`."
    return torch.abs(targ - pred).mean()


def mse(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean squared error between `pred` and `targ`."
    diff = (targ - pred) ** 2
    return diff.mean()


def rmse(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Root mean squared error between `pred` and `targ`."
    return torch.sqrt(mse(pred, targ))


def explained_variance(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Explained variance between `pred` and `targ`."
    var_pct = torch.var(targ - pred) / torch.var(targ)
    return 1 - var_pct


def msle(pred: Tensor, targ: Tensor)->Rank0Tensor:
    "Mean squared logarithmic error between `pred` and `targ`."
    targ = torch.log(1 + targ)
    pred = torch.log(1 + pred)
    diff = (targ - pred) ** 2
    return diff.mean()


def r2_score(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "R2 score (coefficient of determination) between `pred` and `targ`."
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - u / d
