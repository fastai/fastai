import pytest,fastai
from utils.fakes import *
from utils.text import *

## filename: test_train.py
## tests code in train.py

## run: pytest tests/test_train.py (add -s for more screenoutput) 

## see documentation: https://docs.fast.ai/train.html
 
## Model fitting methods     

## To check: is there a more meaningful way to assert, like e.g. fit_one_cycle is 'better' than fit only?
## TO DO: 
  ## summarise this test with test_basic_train test_fit following
  ## https://github.com/fastai/fastai/blob/master/tests/test_callback.py#L65

## fit_one_cycle tested in test_basic_train

def test_lr_find(capsys):
    learn = fake_learner()
    with CaptureStdout() as cs: learning_rate_max = lr_find(learn, start_lr=1e-07, end_lr=10, num_it=100, stop_div=True)
    ## TO DO: consider a dynamic test based on input params
    assert_screenout(cs.out, 'LR Finder is complete')
    
## lr_find tested in other class, e.g. to be added: test_lr_finder 