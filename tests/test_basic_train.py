import pytest,fastai
from utils.fakes import *

## filename: test_basic_train.py
## tests functions in basic_train.py. For test of fit, see test_train

## Class Learner, see documentation:  https://docs.fast.ai/basic_train.html
 
## run: pytest tests/test_basic_train.py (add -s for screenoutput )

def test_get_preds():
    learn = fake_learner()
    a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])