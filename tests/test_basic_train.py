import pytest,fastai
from utils.fakes import *
from utils.text import *

## filename: test_basic_train.py
## tests code in basic_train.py

## run: pytest tests/test_basic_train.py (add -s for more screenoutput)

## Class Learner

## see documentation:  https://docs.fast.ai/basic_train.html
 
## Model fitting methods
 
##TO DO: 
  ## summarise this test with test_train test_fit following
  ## https://github.com/fastai/fastai/blob/master/tests/test_callback.py#L65 


def test_fit(capsys):
    learn = fake_learner()
    learning_rate = 0.01
    weight_decay = 1e-2
    eps = 3
    with CaptureStdout() as cs:  learn.fit(epochs=eps, lr=learning_rate, wd=weight_decay)
    assert_screenout(cs.out, str(eps))
    assert_screenout(cs.out, str('epoch'))

def test_fit_one_cycle(capsys):
    learn = fake_learner()
    cycle_length = 3
    with CaptureStdout() as cs: learn.fit_one_cycle(cyc_len=cycle_length)
    assert_screenout(cs.out, str(cycle_length))
    assert_screenout(cs.out, str('epoch'))

def test_get_preds(capsys):
    learn = fake_learner()
    a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])
   
## TO CHECK: can split method be tested in a useful on linear, fake model or rather with e.g. Unet or Convnet



