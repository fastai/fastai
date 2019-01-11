import pytest,fastai
from utils.fakes import *
from utils.text import *

## filename: test_basic_train.py
## tests code in basic_train.py

## run: pytest tests/test_basic_train.py (add -s for more screenoutput)

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
    captured = capsys.readouterr()
    match_hundperc = re.findall(r'[100%]', captured.out) ## finds 100% progress
    assert match_hundperc, f"expecting to find '100%' in output: f{captured.out}"
      
## fit_one_cycle tested in test_train
## lr_find tested in test_lr_finder

def test_get_preds(capsys):
    learn = fake_learner()
    a = learn.get_preds()
    print ('a: ' + str(a[1]))
    assert learn.data.batch_size == len(a[1])
   
## TO CHECK: can split method be tested in a useful on linear, fake model or rather with e.g. Unet or Convnet



