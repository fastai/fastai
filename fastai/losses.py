from .imports import *
from .torch_imports import *

def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9):
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)
    precision = tp / (y_pred.sum(dim=1)+eps)
    recall = tp / (y_true.sum(dim=1)+eps)
    return torch.mean(
        precision*recall / (precision*(beta**2)+recall+eps) * (1+beta**2))

