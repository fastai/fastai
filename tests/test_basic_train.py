"""
module: basic_train.py - Model fitting methods
docs  : https://docs.fast.ai/train.html
"""

import pytest, fastai
from utils.fakes import *
from utils.text import *

def test_get_preds():
    learn = fake_learner()
    a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])
