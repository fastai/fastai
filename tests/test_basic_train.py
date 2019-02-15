"""
module: basic_train.py - Model fitting methods
docs  : https://docs.fast.ai/train.html
"""

import pytest, fastai
from fastai.vision import *
from utils.fakes import *
from utils.text import *
from fastai.utils.mem import *
from math import isclose

use_gpu = torch.cuda.is_available()

@pytest.fixture(scope="module")
def learn_large_fit():
    preload_pytorch()
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    learn.fit_one_cycle(1)
    return learn

def test_get_preds():
    learn = fake_learner()
    a = learn.get_preds()
    assert learn.data.batch_size == len(a[1])

# WIP
def check_mem(type, expected, received, abs_tol=2):
    assert isclose(expected, received, abs_tol=abs_tol), f"{type} mem: expected={expected} received={received}"

def check_mem_expected(used_expected, peaked_expected, used_received, peaked_received, abs_tol=2):
    assert isclose(used_expected,   used_received,   abs_tol=abs_tol), f"used mem: expected={used_expected} received={used_received}"
    assert isclose(peaked_expected, peaked_received, abs_tol=abs_tol), f"peaked mem: expected={peaked_expected} received={peaked_received}"

@pytest.mark.skipif(not use_gpu, reason="requires cuda")
def test_save_load(learn_large_fit):
    learn = learn_large_fit
    name = 'mnist-tiny-test-save-load'

    with GPUMemTrace() as mem_trace:
        model_path = learn.save(name, return_path=True)
    used_real, peak_real = mem_trace.data()
    check_mem("used", 0, used_real, 5)
    check_mem("peak", 0, peak_real, 5)

    with GPUMemTrace() as mem_trace:
        _ = learn.load(name, purge=True)
    # XXX: very different numbers if done w/o fit first
    check_mem_expected(18, 12, *mem_trace.data(), 5)

    if False:
        # XXX: deep recursion when called after save+load
        with GPUMemTrace() as mem_trace:
            _ = learn.purge()
            _ = learn.load(name)
        check_mem_expected(10, 12, *mem_trace.data(), 5)

    if os.path.exists(model_path): os.remove(model_path)


# hibernate
# destroy
# purge
