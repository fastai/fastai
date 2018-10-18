"Brings TTA (Test Time Functionality) to the `Learner` class. Use `learner.TTA()` instead"
from ..torch_core import *
from ..basic_train import *
from .transform import *

__all__ = ['pred_batch']

def pred_batch(learn:Learner, is_valid:bool=True) -> Tuple[Tensors, Tensors, Tensors]:
    "Returns input, target and output of the model on a batch"
    x,y = next(iter(learn.data.valid_dl if is_valid else learn.data.train_dl))
    return x,y,learn.model(x).detach()
Learner.pred_batch = pred_batch

def _tta_only(learn:Learner, is_test:bool=False, scale:float=1.35) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.data.holdout(is_test)
    ds = dl.dataset
    old = ds.tfms
    augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in
               (crop_pad, flip_lr, dihedral, zoom)]
    try:
        pbar = master_bar(range(8))
        for i in pbar:
            row = 1 if i&1 else 0
            col = 1 if i&2 else 0
            flip = i&4
            d = {'row_pct':row, 'col_pct':col, 'is_random':False}
            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
            if flip: tfm.append(flip_lr(p=1.))
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar)[0]
    finally: ds.tfms = old

Learner.tta_only = _tta_only

def _TTA(learn:Learner, beta:float=0.4, scale:float=1.35, is_test:bool=False) -> Tensors:
    preds,y = learn.get_preds(is_test)
    all_preds = list(learn.tta_only(scale=scale, is_test=is_test))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:            return preds*beta + avg_preds*(1-beta), y

Learner.TTA = _TTA
