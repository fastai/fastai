import pytest,fastai
from fakes import *

## filename: test_lr_finder.py
## tests code in lr_finder.py

## run: pytest tests/test_lr_finder.py -s

## see documentation: 
    #   https://docs.fast.ai/train.html#lr_find 
    #   https://docs.fast.ai/callbacks.lr_finder.html#LRFinder 
 
### tested in class test_lr_finder LRFinder
##def test_lr_find(capsys):
##    learn = fake_learner()
##    learn.lr_find(start_lr=1e-07, end_lr=10, num_it=100, stop_div=True) ## class train.py as entry point
