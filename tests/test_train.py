"""
module: train.py - Model fitting methods
docs  : https://docs.fast.ai/train.html
"""

import pytest, fastai
from utils.fakes import *
from utils.text import *
from fastai.gen_doc.doctest import this_tests

@pytest.fixture(scope="module")
def learn():
    learn = fake_learner(50,50)
    return learn

def test_lr_find(learn):
    this_tests(learn.lr_find)
    wd, start_lr, num_it, end_lr = 0.002, 1e-06, 90, 10
    lr_find(learn=learn, start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=True, wd=wd)
    assert len(learn.recorder.moms) == len(learn.recorder.lrs)
    assert learn.recorder.lrs[0] == start_lr
    assert learn.recorder.moms[0] == 0.9
    assert learn.recorder.lrs[-1] < learn.recorder.opt.lr
    assert learn.recorder.opt.wd == wd
    lr_find(learn=learn, start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=False, wd=wd)
    assert len(learn.recorder.lrs) == num_it

def test_fit(learn):
    this_tests(learn.fit)
    # Test confirms learning rate and momentum are stable, see difference to test_fit_one_cycle
    learning_rate, weight_decay, eps = 3e-3, 1e-2,  4
    with CaptureStdout() as cs:  learn.fit(epochs=eps, lr=learning_rate, wd=weight_decay)
    assert set(learn.recorder.lrs) == {learning_rate}
    assert set(learn.recorder.moms) == {learn.recorder.moms[0]}

def test_fit_one_cycle(learn):
    # Test confirms expected behavior change of learning rate and momentum
    # see graphical representation here: output cell 17 of, learn.sched.plot_lr() in
    # https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb
    lr, cycle_length = 3e-3,  4
    with CaptureStdout() as cs: learn.fit_one_cycle(cycle_length, lr)
    this_tests(learn.fit_one_cycle)
    listlrs = list(learn.recorder.lrs)
    listmoms = list(learn.recorder.moms)
    # we confirm learning rate is at its max when momentum is at its min
    val_lr, idx_lr = max((val, idx) for (idx, val) in enumerate(listlrs))
    val_mom, idx_mom = min((val, idx) for (idx, val) in enumerate(listmoms))
    assert idx_lr == idx_mom
    maxlr_minmom = idx_lr # = idx_mom
    # confirm 1st half (left): learning rate is at its minimum when momentum is at its maximum
    val_lr, idx_lr = min((val, idx) for (idx, val) in enumerate(listlrs[0:maxlr_minmom+1]))
    val_mom, idx_mom = max((val, idx) for (idx, val) in enumerate(listmoms[0:maxlr_minmom+1]))
    assert idx_lr == idx_mom
    # confirm 2nd half (right): learning rate is at its minimum when momentum is at its maximum
    val_lr, idx_lr = min((val, idx) for (idx, val) in enumerate(listlrs[maxlr_minmom:]))
    val_mom, idx_mom = max((val, idx) for (idx, val) in enumerate(listmoms[maxlr_minmom:]))
    assert idx_lr == idx_mom
