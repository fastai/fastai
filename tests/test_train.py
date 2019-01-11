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
def test_fit_one_cycle(capsys):
    learn = fake_learner()
    learning_rate = 0.01
    weight_decay = 1e-2
    learn.fit_one_cycle(cyc_len=3)
    captured = capsys.readouterr()
    match_hundperc = re.findall(r'[100%]', captured.out) ## finds 100% progress
    assert match_hundperc, f"expecting to find '100%' in output: f{captured.out}"

## lr_find tested in other class, e.g. to be added: test_lr_finder 