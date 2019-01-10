import pytest,fastai
from fakes import *

## filename: test_train.py
## tests code in train.py

## run: pytest tests/test_train.py -s

## see documentation: https://docs.fast.ai/train.html
 
## Model fitting methods     

## To check: is there a more meaningful way to assert, like e.g. fit_one_cycle is 'better' than fit only?
def test_fit_one_cycle(capsys):
    learn = fake_learner()
    learning_rate = 0.01
    weight_decay = 1e-2
    learn.fit_one_cycle(cyc_len=3)
    ## TO CHECK: good idea to assert over screen output?   
    captured = capsys.readouterr()
    ##match_epoch = re.findall(r'1/3 ', captured.out) ## finds Epoch 1/3
    ##assert match_epoch
    match_hundperc = re.findall(r'100.00%', captured.out) ## finds 100% progress
    assert match_hundperc

## lr_find tested in class test_lr_finder 