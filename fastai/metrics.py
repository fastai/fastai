"Implements various metrics to measure training accuracy"
from .torch_core import *
from .callback import *

__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'fbeta','Fbeta_binary', 'mse', 'mean_squared_error',
            'mae', 'mean_absolute_error', 'rmse', 'root_mean_squared_error', 'msle', 'mean_squared_logarithmic_error', 
            'explained_variance', 'r2_score', 'balanced_accuracy', 'top_k_accuracy', 'kappa_score', 'confusion_matrix',
            'fbeta_score', 'matthews_corrcoef', 'precision', 'recall']


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


def fbeta_score(input:Tensor, targs:Tensor, beta:float=1, average:bool=False, eps:float=1e-9):
    "If average is True return the unweighted mean of f1 scores across all classes"
    beta = beta ** 2
    cm = confusion_matrix(input, targs)
    prec = torch.diag(cm).float() / cm.sum(0, dtype=torch.float32)
    rec = torch.diag(cm).float() / cm.sum(1, dtype=torch.float32)
    fb = (1 + beta) * (prec * rec) / ((prec * beta) + rec + eps)
    if average:
        return fb.mean()
    else:
        return fb


def confusion_matrix(input:Tensor, targs:Tensor):
    "Computes the confusion matrix."
    if targs.shape == input.shape:
        targs = targs.argmax(-1).view(-1)
    x = torch.arange(0, input.shape[-1])
    input = input.argmax(dim=-1).view(-1)
    cm = ((input==x[:, None]) & (targs==x[:, None, None])).sum(2)
    return cm


def precision(input:Tensor, targs:Tensor, average:bool=False):
    "If average is True return the unweighted mean of precision across all classes"
    cm = confusion_matrix(input, targs)
    prec = torch.diag(cm).float() / cm.sum(0, dtype=torch.float32)
    if average:
        return prec.mean()
    else:
        return prec


def recall(input:Tensor, targs:Tensor, average:bool=False):
    "If average is True return the unweighted mean of recall across all classes"
    cm = confusion_matrix(input, targs)
    rec = torch.diag(cm).float() / cm.sum(1, dtype=torch.float32)
    if average:
        return rec.mean()
    else:
        return rec



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


def balanced_accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    """
    Accuracy score between `input` and `targs` based on class label frequency. 
    Defined as the unweighted mean of recall for each class.
    """
    return recall(input, targs, average=True)


def top_k_accuracy(input:Tensor, targs:Tensor, k:int=5)->Rank0Tensor:
    "Computes the Top-k accuracy (target is in the top k predictions)."
    n = targs.shape[0]
    input = input.topk(k=k, dim=-1)[1].view(n, -1)
    targs = targs.view(n,-1)
    return (input == targs).sum(dim=1, dtype=torch.float32).mean()


def mean_class_accuracy(input: Tensor, targs: Tensor):
    "Computes the accuracy for each class label ->Rank1Tensor"
    x = torch.arange(0, input.shape[-1])
    targs = targs==x[:,None]
    input = input.argmax(-1)==x[:,None]
    label_sum = targs.sum(dim=1).float()
    eq = targs.float() * input.float()
    return eq.sum(1) / label_sum


def matthews_corrcoef(input:Tensor, targs:Tensor):
    """
    Matthews correlation coefficient.
    Ref: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py
    """
    cm = confusion_matrix(input, targs)
    t_sum = cm.sum(dim=1, dtype=torch.float32)
    p_sum = cm.sum(dim=0, dtype=torch.float32)
    n_correct = torch.trace(cm).float()
    n_samples = p_sum.sum(dtype=torch.float32)
    cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - torch.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - torch.dot(t_sum, t_sum)
    mcc = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)
    return mcc



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
    """
    Computes the rate of agreement (Cohens Kappa) between `pred` and `rater`..
    Ref: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py
    """
    n = pred.shape[-1]
    cm = confusion_matrix(pred, rater).float()
    sum0 = cm.sum(0)
    sum1 = cm.sum(1)
    expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
    w = torch.ones((n, n))
    idx = torch.arange(0, n)
    w[idx, idx] = 0
    k = torch.sum(w * cm) / torch.sum(w * expected)
    return 1 - k


def exp_rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    pred, targ = torch.exp(pred), torch.exp(targ)
    pct_var = (targ - pred)/targ
    return torch.sqrt((pct_var**2).mean())


def mean_absolute_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean absolute error between `pred` and `targ`."
    return torch.abs(targ - pred).mean()


def mean_squared_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean squared error between `pred` and `targ`."
    return F.mse_loss(pred, targ)


def root_mean_squared_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Root mean squared error between `pred` and `targ`."
    return torch.sqrt(F.mse_loss(pred, targ))


def mean_squared_logarithmic_error(pred: Tensor, targ: Tensor)->Rank0Tensor:
    "Mean squared logarithmic error between `pred` and `targ`."
    return F.mse_loss(torch.log(1 + pred), torch.log(1 + targ))


def explained_variance(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Explained variance between `pred` and `targ`."
    var_pct = torch.var(targ - pred) / torch.var(targ)
    return 1 - var_pct


def r2_score(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "R2 score (coefficient of determination) between `pred` and `targ`."
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - u / d


# Aliases

mse = mean_squared_error
mae = mean_absolute_error
msle = mean_squared_logarithmic_error
rmse = root_mean_squared_error


@dataclass
class Fbeta_binary(Callback):
    "Computes the fbeta between preds and targets for single-label classification"
    beta2: int = 2
    eps: float = 1e-9
    clas:int=1
    
    def on_epoch_begin(self, **kwargs):
        self.TP = 0
        self.total_y_pred = 0   
        self.total_y_true = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        y_pred = last_output.argmax(dim=1)
        y_true = last_target.float()
        
        self.TP += ((y_pred==clas) * (y_true==clas)).float().sum()
        self.total_y_pred += (y_pred==clas).float().sum()
        self.total_y_true += (y_true==clas).float().sum()
    
    def on_epoch_end(self, **kwargs):
        beta2=self.beta2**2
        prec = self.TP/(self.total_y_pred+self.eps)
        rec = self.TP/(self.total_y_true+self.eps)       
        res = (prec*rec)/(prec*beta2+rec+self.eps)*(1+beta2)
        self.metric = res 
