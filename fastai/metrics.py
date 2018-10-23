"Implements various metrics to measure training accuracy"
from .torch_core import *
from .callback import *

__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'Fbeta']

@dataclass
class Fbeta(Callback):
    thresh:float=0.5
    beta:float=2
    eps:float=1e-9
    sigmoid:bool=True
    
    def on_epoch_begin(self, **kwargs):
        self.TP, self.pred, self.true = 0, 0, 0
    
    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if self.sigmoid: last_output = last_output.sigmoid()
        y_pred = (last_output>self.thresh).float()
        y_true = last_target.float()
        self.TP += (y_pred*y_true).sum(dim=0)
        self.pred += y_pred.sum(dim=0)
        self.true += y_true.sum(dim=0)
    
    def on_epoch_end(self, **kwargs):
        beta2 = self.beta**2
        prec = self.TP/(self.pred+self.eps)
        rec = self.TP/(self.true+self.eps)
        res = (prec*rec)/(prec*beta2+rec+self.eps)*(1+beta2)
        self.metric = res.mean().detach().item()

def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, beta:float=2, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:
    "Compute the f_beta between preds and targets."
    beta2 = beta**2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()[:,None]
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean()

def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return 2. * intersect / union
    else: return intersect / (union-intersect+1.0)

def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def error_rate(input:Tensor, targs:Tensor)->Rank0Tensor:
    "1 - `accuracy`"
    return 1-accuracy(input, targs)

def exp_rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())
