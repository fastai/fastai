import pytest,fastai
from utils.fakes import *
from utils.text import *

## filename: test_train.py
## tests functions in train.py

## Class Learner, see documentation:  https://docs.fast.ai/train.html
 
## Model fitting methods

## run: pytest tests/test_train.py (add -s for screenoutput )


@pytest.fixture(scope="module")
def learn():
    
    learn = fake_learner(50,50)
   
    return learn

def test_fit(learn):

    ## Test confirms learning rate and momentum are stable, see difference to test_fit_one_cycle
    
    learning_rate = 3e-3
    weight_decay = 1e-2
    eps = 4
    
    with CaptureStdout() as cs:  learn.fit(epochs=eps, lr=learning_rate, wd=weight_decay)
    #assert_screenout(cs.out, str(eps))
    lrs = list(learn.recorder.lrs)
    prevlr = learning_rate
    for lr in lrs:
        if prevlr is not None: assert prevlr == lr
        prevlr = lr
    moms = list(learn.recorder.moms)
    prevmom = None
    for mom in moms:
        if prevmom is not None: assert prevmom == mom
        prevmom = mom
   
def test_fit_one_cycle(learn):

    ## Test confirms expected behavior leaning rate and momentum
    ## see graphical representation here: output cell 17 of, learn.sched.plot_lr() in 
    ## https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb

    cycle_length = 4
    lr = 3e-3 # 5e-3

    with CaptureStdout() as cs: learn.fit_one_cycle(cycle_length, lr)
    #assert_screenout(cs.out, str(cycle_length)) 

    listlrs = list(learn.recorder.lrs)
    listmoms = list(learn.recorder.moms) # give give_moms(learn)
    
    ## eliminate the final 'off' lrs
    for (idx,lr) in enumerate(listlrs):
        if lr < listlrs[0]:
            del listlrs[idx]
            del listmoms[idx]

    ## we confirm learning rate is at its max when momentum is at its low
    val_lr, idx_lr = max((val, idx) for (idx, val) in enumerate(listlrs))
    val_mom, idx_mom = min((val, idx) for (idx, val) in enumerate(listmoms))
    assert idx_lr == idx_mom

    ## we separate the graph 1st half (left) and 2nd part (right) with previously identified saddle point
    maxlr_minmom = idx_lr # = idx_mom
    
    ## confirm 1st half (left): learning rate is at its minimum when momentum is at its maximum
    val_lr, idx_lr = min((val, idx) for (idx, val) in enumerate(listlrs[0:maxlr_minmom+1]))
    val_mom, idx_mom = max((val, idx) for (idx, val) in enumerate(listmoms[0:maxlr_minmom+1]))
    assert idx_lr == idx_mom

    ## confirm 2nd half (right): learning rate is at its minimum when momentum is at its maximum
    ## Note, for simplicity we cut off the final learning rates < start learning rate and ignore the final stable momentum
    val_lr, idx_lr = min((val, idx) for (idx, val) in enumerate(listlrs[maxlr_minmom:])) ## remove previous min.
    val_mom, idx_mom = max((val, idx) for (idx, val) in enumerate(listmoms[maxlr_minmom:])) ## remove previous max.
   
    assert idx_lr == idx_mom

    
