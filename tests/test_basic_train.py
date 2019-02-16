"""
module: basic_train.py - Model fitting methods
docs  : https://docs.fast.ai/train.html
"""

import pytest, fastai
from fastai.vision import *
from utils.fakes import *
from utils.text import *
from utils.mem import *
from fastai.utils.mem import *
from math import isclose

torch_preload_mem()

# this is not a fixture on purpose - the memory measurement tests are very
# fickle, so they need a fresh object and not one modified by other tests
def learn_large_unfit():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    return learn

@pytest.fixture(scope="module")
def learn(): return learn_large_unfit()

def test_get_preds():
    learn = fake_learner()
    with CaptureStdout() as cs:
        a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])

def test_save_load(learn):
    name = 'mnist-tiny-test-save-load'

    # testing that all these various sequences don't break each other
    model_path = learn.save(name, return_path=True)
    _ = learn.load(name, purge=True)
    learn.data.sanity_check()
    assert 709 == len(learn.data.train_ds)
    _ = learn.purge()
    _ = learn.load(name)
    _ = learn.load(name)
    model_path = learn.save(name, return_path=True)
    _ = learn.load(name, purge=True)
    # basic checks
    #assert learn.recorder
    assert learn.opt
    assert 709 == len(learn.data.train_ds)
    # XXX: could use more sanity checks

    if os.path.exists(model_path): os.remove(model_path)

def check_mem_expected(used_expected, peaked_expected, mtrace, abs_tol=2):
    used_received, peaked_received = mtrace.data()
    assert isclose(used_expected,   used_received,   abs_tol=abs_tol), f"used mem: expected={used_expected} received={used_received}"
    assert isclose(peaked_expected, peaked_received, abs_tol=abs_tol), f"peaked mem: expected={peaked_expected} received={peaked_received}"

#@pytest.mark.skip(reason="WIP")
@pytest.mark.cuda
def test_save_load_mem_leak():
    learn = learn_large_unfit()
    name = 'mnist-tiny-test-save-load'
    #learn.fit_one_cycle(1)

    # save should consume no extra used or peaked memory
    with GPUMemTrace() as mtrace:
        model_path = learn.save(name, return_path=True)
    check_mem_expected(used_expected=0, peaked_expected=0, mtrace=mtrace, abs_tol=10)

    # load w/ purge still leaks some the first time it's run
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
    # XXX: very different numbers if done w/o fit first 42 8, w/ fit 24 16
    check_mem_expected(used_expected=42, peaked_expected=8, mtrace=mtrace, abs_tol=10)

    # subsequent multiple load w/o purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=False)
        learn.load(name, purge=False)
    check_mem_expected(used_expected=0, peaked_expected=20, mtrace=mtrace, abs_tol=10)

    # subsequent multiple load w/ purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
        learn.load(name, purge=True)
    check_mem_expected(used_expected=0, peaked_expected=20, mtrace=mtrace, abs_tol=10)

    # purge + load w/ default purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.purge()
        learn.load(name)
    check_mem_expected(used_expected=0, peaked_expected=20, mtrace=mtrace, abs_tol=10)

    if os.path.exists(model_path): os.remove(model_path)


# hibernate
# destroy
# purge
