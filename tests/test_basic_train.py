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

@pytest.fixture(scope="module")
def data():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    return data

# this is not a fixture on purpose - the memory measurement tests are very sensitive, so
# they need to be able to get a fresh learn object and not one modified by other tests.
def learn_large_unfit(data):
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    return learn

@pytest.fixture(scope="module")
def learn(data): return learn_large_unfit(data)

def test_get_preds():
    learn = fake_learner()
    with CaptureStdout() as cs:
        a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])

def test_save_load(learn):
    name = 'mnist-tiny-test-save-load'

    # testing that all these various sequences don't break each other
    model_path = learn.save(name, return_path=True)
    learn.load(name, purge=True)
    learn.data.sanity_check()
    assert 709 == len(learn.data.train_ds)
    learn.purge()
    learn.load(name)
    learn.load(name)
    model_path = learn.save(name, return_path=True)
    learn.load(name, purge=True)
    # basic checks
    #assert learn.recorder
    assert learn.opt
    assert 709 == len(learn.data.train_ds)
    # XXX: could use more sanity checks

    if os.path.exists(model_path): os.remove(model_path)

def check_mem_expected(used_exp, peaked_exp, mtrace, abs_tol=2, ctx=None):
    used_received, peaked_received = mtrace.data()
    ctx = f" ({ctx})" if ctx is not None else ""
    assert isclose(used_exp,   used_received,   abs_tol=abs_tol), f"used mem: expected={used_exp} received={used_received}{ctx}"
    assert isclose(peaked_exp, peaked_received, abs_tol=abs_tol), f"peaked mem: expected={peaked_exp} received={peaked_received}{ctx}"

def report_mem_real(used_exp, peaked_exp, mtrace, abs_tol=2, ctx=None):
    ctx = f" ({ctx})" if ctx is not None else ""
    print(f"{mtrace}{ctx}")
#check_mem_expected = report_mem_real

#@pytest.mark.skip(reason="WIP")
@pytest.mark.cuda
def test_save_load_mem_leak(data):
    learn = learn_large_unfit(data)
    name = 'mnist-tiny-test-save-load'
    #learn.fit_one_cycle(1)

    # A big difficulty with measuring memory consumption is that it varies quite
    # wildly from one GPU model to another.
    #
    # Perhaps we need sets of different expected numbers per developer's GPUs?
    # override check_mem_expected above with report_mem_real to acquire a new set
    #
    # So for now just testing the specific card I have until a better way is found.
    dev_name = torch.cuda.get_device_name(None)
    if dev_name != 'GeForce GTX 1070 Ti':
        pytest.skip(f"currently only matched for mem usage on specific GPU models, {dev_name} is not one of them")

    # save should consume no extra used or peaked memory
    with GPUMemTrace() as mtrace:
        model_path = learn.save(name, return_path=True)
    check_mem_expected(used_exp=0, peaked_exp=0, mtrace=mtrace, abs_tol=10, ctx="save")

    # load w/ purge still leaks some the first time it's run
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
    # XXX: very different numbers if done w/o fit first 42 8, w/ fit 24 16
    check_mem_expected(used_exp=42, peaked_exp=8, mtrace=mtrace, abs_tol=10, ctx="load")

    # subsequent multiple load w/o purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=False)
        learn.load(name, purge=False)
    check_mem_expected(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="load x 2")

    # subsequent multiple load w/ purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
        learn.load(name, purge=True)
    check_mem_expected(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="load x 2 2nd time")

    # purge + load w/ default purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.purge()
        learn.load(name)
    check_mem_expected(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="purge+load")

    if os.path.exists(model_path): os.remove(model_path)
