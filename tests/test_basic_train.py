"""
module: basic_train.py - Model fitting methods
docs  : https://docs.fast.ai/train.html
"""

import pytest
from fastai.vision import *
from fastai.utils.mem import *
from fastai.gen_doc.doctest import this_tests
from utils.fakes import *
from utils.text import *
from utils.mem import *
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
    this_tests(learn.get_preds)
    with CaptureStdout() as cs:
        a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])

def test_freeze_to():
    learn = fake_learner(layer_group_count=3)
    this_tests(learn.freeze_to)
    learn.freeze_to(1)
    for i, param in enumerate(learn.model.parameters()):
        # param 0 is weights in layer_group 0 and param 1 is bias in layer_group 0
        # all params other than those should be frozen
        if i >= 2: assert param.requires_grad == True
        else:      assert param.requires_grad == False

def test_freeze():
    learn = fake_learner(layer_group_count=3)
    this_tests(learn.freeze)
    learn.freeze()
    for i, param in enumerate(learn.model.parameters()):
        # 2 layer groups with 1 param in each should be frozen
        if i >= 4: assert param.requires_grad == True
        else:      assert param.requires_grad == False

def test_unfreeze():
    learn = fake_learner(layer_group_count=4)
    this_tests(learn.unfreeze)
    for param in learn.model.parameters(): param.requires_grad=False
    learn.unfreeze()
    for param in learn.model.parameters(): assert param.requires_grad == True

def check_learner(learn, train_items):
    # basic checks
    #assert learn.recorder
    assert learn.model
    assert train_items == len(learn.data.train_ds.items)
    # XXX: could use more sanity checks

def test_save_load(learn):
    this_tests(learn.save, learn.load)
    name = 'mnist-tiny-test-save-load'
    train_items = len(learn.data.train_ds)
    # testing that all these various sequences don't break each other
    model_path = learn.save(name, return_path=True)
    learn.load(name, purge=True)
    learn.data.sanity_check()
    check_learner(learn, train_items)

    learn.purge()
    learn.load(name)
    learn.load(name)
    model_path = learn.save(name, return_path=True)
    learn.load(name, purge=True)
    check_learner(learn, train_items)
    if os.path.exists(model_path): os.remove(model_path)

def subtest_save_load_mem(data):
    learn = learn_large_unfit(data)
    name = 'mnist-tiny-test-save-load'
    #learn.fit_one_cycle(1)

    # save should consume no extra used or peaked memory
    with GPUMemTrace() as mtrace:
        model_path = learn.save(name, return_path=True)
    check_mtrace(used_exp=0, peaked_exp=0, mtrace=mtrace, abs_tol=10, ctx="save")

    # load w/ purge still leaks some the first time it's run
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
    # XXX: very different numbers if done w/o fit first 42 8, w/ fit 24 16
    check_mtrace(used_exp=42, peaked_exp=8, mtrace=mtrace, abs_tol=10, ctx="load")

    # subsequent multiple load w/o purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=False)
        learn.load(name, purge=False)
    check_mtrace(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="load x 2")

    # subsequent multiple load w/ purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.load(name, purge=True)
        learn.load(name, purge=True)
    check_mtrace(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="load x 2 2nd time")

    # purge + load w/ default purge should consume no extra used memory
    with GPUMemTrace() as mtrace:
        learn.purge()
        learn.load(name)
    check_mtrace(used_exp=0, peaked_exp=20, mtrace=mtrace, abs_tol=10, ctx="purge+load")

    if os.path.exists(model_path): os.remove(model_path)

def test_destroy():
    msg = "this object has been destroyed"
    learn = fake_learner()
    this_tests(learn.destroy)
    with CaptureStdout() as cs: learn.destroy()
    assert "this Learner object self-destroyed" in cs.out

    # should be able to re-run learn.destroy multiple times for nb convenience
    with CaptureStdout() as cs: learn.destroy()
    assert msg in cs.out

    # should be able to run normal methods, except they are no-ops and say that they are
    with CaptureStdout() as cs: learn.fit(1)
    assert msg in cs.out

    # should be able to call attributes, except they are gone and say so
    # unless they are __getattr__' loaded from Learner, in which case they are still normal
    for attr in ['data', 'model', 'callbacks']:
        with CaptureStdout() as cs: val = getattr(learn, attr, None)
        assert msg in cs.out, attr
        assert val is None, attr

    # check that `destroy` didn't break the Learner class
    learn = fake_learner()
    with CaptureStdout() as cs: learn.fit(1)
    assert "epoch" in cs.out
    assert "train_loss" in cs.out


def subtest_destroy_mem(data):
    with GPUMemTrace() as mtrace:
        learn = learn_large_unfit(data)
    check_mtrace(used_exp=20, peaked_exp=0, mtrace=mtrace, abs_tol=10, ctx="load")

    # destroy should free most of the memory that was allocated during load (training, etc.)
    with GPUMemTrace() as mtrace:
        with CaptureStdout() as cs: learn.destroy()
    check_mtrace(used_exp=-20, peaked_exp=0, mtrace=mtrace, abs_tol=10, ctx="destroy")

# memory tests behave differently when run individually and in a row, since
# memory utilization patterns are very inconsistent - would require a full gpu
# card reset before each test to be able to test consistently, so will run them
# all in a precise sequence
@pytest.mark.cuda
def test_memory(data):
    # A big difficulty with measuring memory consumption is that it varies quite
    # wildly from one GPU model to another.
    #
    # Perhaps we need sets of different expected numbers per developer's GPUs?
    # override check_mem above in tests.utils.mem with report_mem to acquire a new set
    #
    # So for now just testing the specific card I have until a better way is found.

    dev_name = torch.cuda.get_device_name(None)
    if dev_name != 'GeForce GTX 1070 Ti':
        pytest.skip(f"currently only matched for mem usage on specific GPU models, {dev_name} is not one of them")

    subtest_save_load_mem(data)
    subtest_destroy_mem(data)

def test_export_load_learner():
    export_file = 'export.pkl'
    for should_destroy in [False, True]:
        learn = fake_learner()
        this_tests(learn.export, load_learner)
        path = learn.path
        with CaptureStdout() as cs: learn.export(destroy=should_destroy)
        learn = load_learner(path)
        # export removes data
        train_items = 0
        check_learner(learn, train_items)
        if os.path.exists(export_file): os.remove(export_file)
