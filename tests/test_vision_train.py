import pytest
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from utils.mem import *
from io import StringIO
from contextlib import redirect_stdout
from math import isclose

use_gpu = can_use_gpu()
torch_preload_mem()

pytestmark = pytest.mark.integration

@pytest.fixture
def no_bar():
    fastprogress.NO_BAR = True
    yield
    fastprogress.NO_BAR = False

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), num_workers=2)
    data.normalize()
    learn = Learner(data, simple_cnn((3,16,16,16,2), bn=True), metrics=[accuracy, error_rate])
    learn.fit_one_cycle(3)
    return learn

@pytest.fixture(scope="module")
def learn_large_unfit():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    return create_cnn(data, models.resnet18, metrics=accuracy)

def test_accuracy(learn):
    assert accuracy(*learn.get_preds()) > 0.9

def test_error_rate(learn):
    assert error_rate(*learn.get_preds()) < 0.1

def test_1cycle_lrs(learn):
    lrs = learn.recorder.lrs
    assert lrs[0]<0.001
    assert lrs[-1]<0.0001
    assert np.max(lrs)==3e-3

def test_1cycle_moms(learn):
    moms = learn.recorder.moms
    assert moms[0]==0.95
    assert abs(moms[-1]-0.95)<0.01
    assert np.min(moms)==0.85

def test_preds(learn):
    pass_tst = False
    for i in range(3):
        img, label = learn.data.valid_ds[i]
        pred_class,pred_idx,outputs = learn.predict(img)
        if outputs[int(label)] > outputs[1-int(label)]: return
    assert False, 'Failed to predict correct class'

def test_interp(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_interp_shortcut(learn):
    interp = learn.interpret()
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_lrfind(learn):
    learn.lr_find(start_lr=1e-5,end_lr=1e-3, num_it=15)

def test_model_save_load(learn):
    "testing save/load cycle"

    summary_before = model_summary(learn)
    name = 'mnist-tiny-test-save-load'
    model_path = learn.save(name=name, return_path=True)
    _ = learn.load(name)
    if os.path.exists(model_path): os.remove(model_path)
    summary_after = model_summary(learn)
    assert summary_before == summary_after, f"model summary before and after"

def test_model_load_mem_leak(learn_large_unfit):
    "testing memory leak on load"

    pytest.xfail("memory leak in learn.load()")

    learn = learn_large_unfit
    gpu_mem_reclaim() # baseline
    used_before = gpu_mem_get_used()

    name = 'mnist-tiny-test-load-mem-leak'
    model_path = learn.save(name=name, return_path=True)
    _ = learn.load(name)
    if os.path.exists(model_path): os.remove(model_path)
    used_after = gpu_mem_get_used()

    # models.resnet18 loaded in GPU RAM is about 50MB
    # calling learn.load() of a saved and then instantly re-loaded model shouldn't require more GPU RAM
    # XXX: currently w/o running gc.collect() this temporarily leaks memory and causes fragmentation - the fragmentation can't be tested from here, but it'll get automatically fixed once load is fixed. load() must unload first the previous model, gc.collect() and only then load the new one onto cuda.
    assert isclose(used_before, used_after, abs_tol=6), f"load() and used GPU RAM: before load(): {used_before}, after: {used_after}"

    # this shows how it should have been
    gc.collect()
    gpu_cache_clear()
    used_after_reclaimed = gpu_mem_get_used()
    # XXX: not sure where 6MB get lost still but for now it's a small leak - need to test with a bigger model
    assert isclose(used_before, used_after_reclaimed, abs_tol=6),f"load() and used GPU RAM: before load(): {used_before}, after: {used_after}, after gc.collect() {used_after_reclaimed} used"
