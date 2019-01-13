"Implements various metrics to measure training accuracy"
from .torch_core import *
from .callback import *


__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'fbeta','FBeta', 'mse', 'mean_squared_error',
            'mae', 'mean_absolute_error', 'rmse', 'root_mean_squared_error', 'msle', 'mean_squared_logarithmic_error', 
            'explained_variance', 'r2_score', 'top_k_accuracy', 'KappaScore', 'ConfusionMatrix', 'MatthewsCorreff', 
            'Precision', 'Recall', 'R2Score', 'ExplainedVariance', 'ExpRMSPE', 'RMSE']


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

def top_k_accuracy(input:Tensor, targs:Tensor, k:int=5)->Rank0Tensor:
    "Computes the Top-k accuracy (target is in the top k predictions)."
    n = targs.shape[0]
    input = input.topk(k=k, dim=-1)[1].view(n, -1)
    targs = targs.view(n,-1)
    return (input == targs).sum(dim=1, dtype=torch.float32).mean()

def error_rate(input:Tensor, targs:Tensor)->Rank0Tensor:
    "1 - `accuracy`"
    return 1 - accuracy(input, targs)

def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return intersect / (union-intersect+1.0)

def exp_rmspe(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Exp RMSE between `pred` and `targ`."
    assert pred.numel() == targ.numel(), "Expected same numbers of elements in pred & targ"
    if len(pred.shape)==2: pred=pred.squeeze(1)
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

def mean_squared_logarithmic_error(pred:Tensor, targ:Tensor)->Rank0Tensor:
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


class RegMetrics(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def on_epoch_begin(self):
        self.targs, self.preds = Tensor([]), Tensor([])

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        assert last_output.numel() == last_target.numel(), "Expected same numbers of elements in pred & targ"
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu()))

class R2Score(RegMetrics):
    def on_epoch_end(self):
        self.metric = r2_score(self.preds, self.targs)

class ExplainedVariance(RegMetrics):
    def on_epoch_end(self):
        self.metric = explained_variance(self.preds, self.targs)

class RMSE(RegMetrics):
    def on_epoch_end(self):
        self.metric = root_mean_squared_error(self.preds, self.targs)

class ExpRMSPE(RegMetrics):
    def on_epoch_end(self):
        self.metric = exp_rmspe(self.preds, self.targs)

# Aliases
mse = mean_squared_error
mae = mean_absolute_error
msle = mean_squared_logarithmic_error
rmse = root_mean_squared_error

class ConfusionMatrix(Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm
        
    def on_epoch_end(self, **kwargs):
        self.metric = self.cm
        
        
@dataclass
class CMScores(ConfusionMatrix):
    "Base class for metrics which rely on the calculation of the precision and/or recall score."
    
    average:Optional[str]="binary"      # `binary`, `micro`, `macro`, `weigthed` or None
    pos_label:int=1                     # 0 or 1
    eps:float=1e-9

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        if self.average is None:
            return rec 
        else:
            if self.average == "micro":
                weights = self._weights(avg="weighted")
            else:
                weights = self._weights(avg=self.average)
            return (rec * weights).sum()
 

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)
        if self.average is None:
            return prec 
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()
        

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1:
                return Tensor([0,1])
            else: 
                return Tensor([1,0])
        elif avg == "micro":
            return self.cm.sum(dim=0) / self.cm.sum()
        elif avg == "macro":
            return torch.ones((self.n_classes,)) / self.n_classes
        elif avg == "weighted":
            return self.cm.sum(dim=1) / self.cm.sum()


class Recall(CMScores):
    def on_epoch_end(self, **kwargs):
        self.metric = self._recall()
            

class Precision(CMScores):
    def on_epoch_end(self, **kwargs):
        self.metric = self._precision()
            
            
@dataclass
class FBeta(CMScores):
    beta:float=2
        
    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average
        if self.average != "micro":
            self.average = None

    def on_epoch_end(self, **kwargs):
        prec = self._precision()
        rec = self._recall()
        self.metric = (1 + self.beta2) * prec * rec / (prec * self.beta2 + rec + self.eps)
        if self.avg:
            self.metric = (self._weights(avg=self.avg) * self.metric).sum()
            
    def on_train_end(self, **kwargs): self.average = self.avg

class KappaScore(ConfusionMatrix):
    """
    Computes the rate of agreement (Cohens Kappa).
    Ref.: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py
    """
    
    def on_epoch_end(self, **kwargs):
        w = torch.ones((self.n_classes, self.n_classes))
        w[self.x, self.x] = 0
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        self.metric = 1 - k
        

class MatthewsCorreff(ConfusionMatrix):
    """    
    Computes the Matthews correlation coefficient.
    Ref.: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py
    """

    def on_epoch_end(self, **kwargs):
        t_sum = self.cm.sum(dim=1)
        p_sum = self.cm.sum(dim=0)
        n_correct = torch.trace(self.cm)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - torch.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - torch.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - torch.dot(t_sum, t_sum)
        self.metric = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)
