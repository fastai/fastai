"Brings TTA (Test Time Functionality) to the `Learner` class. Use `learner.TTA()` instead"
from ..torch_core import *
from ..basic_train import *
from ..basic_train import _loss_func2activ
from ..basic_data import DatasetType
from .transform import *

__all__ = []

def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, scale:float=1.35) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
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
            yield get_preds(learn.model, dl, pbar=pbar, activ=_loss_func2activ(learn.loss_func))[0]
    finally: ds.tfms = old

Learner.tta_only = _tta_only

def _TTA(learn:Learner, beta:float=0.4, scale:float=1.35, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type)
    all_preds = list(learn.tta_only(scale=scale, ds_type=ds_type))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:            
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss: 
            return final_preds, y, calc_loss(final_preds, y, learn.loss_func)
        return final_preds, y

Learner.TTA = _TTA
