import pytest,fastai
from fakes import *

## filename: test_basic_train.py
## tests code in basic_train.py

## run: pytest tests/test_basic_train.py -s

## Class Learner

## see documentation:  https://docs.fast.ai/basic_train.html
 
## Model fitting methods
 
def test_fit(capsys):
    learn = fake_learner()
    learning_rate = 0.01
    weight_decay = 1e-2
    learn.fit(epochs=3, lr=learning_rate, wd=weight_decay)
    assert learn.opt.lr == learn.lr_range(learning_rate)
    assert learn.opt.wd == weight_decay
    ## TO CHECK: good idea to assert over this output?    
    captured = capsys.readouterr()
    ##match_epoch = re.findall(r'1/3 ', captured.out) ## finds Epoch 1/3
    ##assert match_epoch
    match_hundperc = re.findall(r'100.00%', captured.out) ## finds 100% progress
    assert match_hundperc
      
## fit_one_cycle tested in test_train
## lr_find tested in test_lr_finder

def test_get_preds(capsys):
    learn = fake_learner()
    print ('learn: ' + str(learn))
    print ('learn.data.batch_size: ' + str(learn.data.batch_size))
    a = learn.get_preds()
    print ('a: ' + str(a[1]))
    assert learn.data.batch_size == len(a[1])
   
## TO CHECK: can split method be tested in a useful on linear, fake model or rather with e.g. Unet or Convnet



