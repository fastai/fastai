import pytest,fastai
from fakes import *

## filename: test_basic_data.py
## tests code in basic_data.py -s

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
      
