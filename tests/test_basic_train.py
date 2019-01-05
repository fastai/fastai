import pytest,fastai
from fakes import *

## filename: test_basic_train.py
## tests code in basic_train.py -s

## run: pytest tests/test_basic_train.py -s

## Class Learner
 
def test_fit(capsys):
    learn = fake_learner()
    learning_rate = 0.001;
    weight_decay = 0.01;
    learn.fit(epochs=3, lr=learning_rate, wd=weight_decay)
    assert learn.opt.lr == learn.lr_range(learning_rate)
    assert learn.opt.wd == weight_decay
    captured = capsys.readouterr()
    match_epoch = re.findall(r'1/3 ', captured.out) ## finds Epoch 1/3
    assert match_epoch
    match_hundperc = re.findall(r'100.00%', captured.out) ## finds 100% progress
    assert match_hundperc
      
## TO DO Class Learner lr_range(self, lr:Union[float,slice])->np.ndarray:
## TO DO Class Learner create_opt(self, lr:Floats, wd:Floats=0.)->None:
## TO DO Class Learner split(self, split_on:SplitFuncOrIdxList)->None:      
## TO DO Class Learner freeze_to(self, n:int)->None:             
## TO DO Class Learner freeze(self)->None:
## TO DO Class Learner unfreeze(self):
## TO DO Class Learner get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None,  pbar:Optional[PBar]=None) -> List[Tensor]:
## TO DO Class Learner pred_batch(self, ds_type:DatasetType=DatasetType.Valid, batch:Tuple=None, reconstruct:bool=False) -> List[Tensor]:
## TO DO Class Learner def backward(self, item): Useful if `backward_hooks` are attached."        
## TO DO Class Learner predict(self, item:ItemBase, **kwargs):
## TO DO Class Learner validate(self, dl=None, callbacks=None, metrics=None):
        
## TO DO: class LearnerCallback(Callback):
## TO DO: class Recorder(LearnerCallback):
## TO DO: class RecordOnCPU(Callback):
  
## TO DO: loss_batch
## TO DO: train_epoch