from .imports import *
from .torch_imports import *

def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds==targs).float().mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh).float()==targs).float().mean()

def accuracy_multi_np(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def recall(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/targs.sum()

def precision(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/pred_pos.sum()

def fbeta(preds, targs, beta, thresh=0.5):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(preds, targs, thresh)
    prec = precision(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec)

def f1(preds, targs, thresh=0.5): return fbeta(preds, targs, 1, thresh)
