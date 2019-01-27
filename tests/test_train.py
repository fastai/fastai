import pytest,fastai
from utils.fakes import *
from utils.text import *
from utils.param import *
from fastai.vision import *

## filename: test_train.py
## tests functions in train.py

## Class Learner, see documentation:  https://docs.fast.ai/train.html
 
## Model fitting methods

## run: pytest tests/test_train.py (add -s for screenoutput )


@pytest.fixture(scope="module")
def learn():
    ## fixture for same model in fit and fit_one_cycle and demonstrate difference in behavior
    ## test_fit and test_fit_one_cycle assert correctly with decommented models, 
    ## fake_learner defaulted for performance reasons
    
    learn = fake_learner(50,50)

    #path = untar_data(URLs.MNIST_TINY)
    #data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), num_workers=2)
    #data.normalize()
    #learn = create_cnn(data, models.resnet34, metrics=error_rate)
    #learn = Learner(data, simple_cnn((3,16,16,16,2), bn=True), metrics=[accuracy, error_rate])        
    
    return learn

def test_fit(learn):

    ## Test confirms learning rate and momentum are stable, see difference to test_fit_one_cycle
    
    learning_rate = 3e-3
    weight_decay = 1e-2
    eps = 4
    
    with CaptureStdout() as cs:  learn.fit(epochs=eps, lr=learning_rate, wd=weight_decay)
    assert_screenout(cs.out, str(eps))
    lrs = get_learning_rates(learn)
    prevlr = learning_rate
    for lr in lrs:
        if prevlr is not None: assert prevlr == lr
        prevlr = lr
    moms = get_momentum(learn)
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
    assert_screenout(cs.out, str(cycle_length)) 

    listlrs =  get_learning_rates(learn) #give_lrs(learn)
    listmoms = get_momentum(learn) # give give_moms(learn)
    
    ## eliminate the final 'off' lrs
    for (idx,lr) in enumerate(listlrs):
        if lr < listlrs[0]:
            del listlrs[idx]
            del listmoms[idx]

    '''
    ## list learning rate and momentum as pairs at the end of each training epoch
    ## for debugging
    listlrsandmoms = get_lrsmoms_paired(learn)
    i=0
    for lrandmom in listlrsandmoms:
        print(' \n\t' + str(i) + ' learning rate: ' + str(lrandmom[0]))
        print(' \t' + str(i) + ' momentum: ' + str(lrandmom[1]))
        i =  i + 1
    '''

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

    
