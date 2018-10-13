import pytest
from fastai import *
from fastai.vision import *

@pytest.mark.slow
@pytest.fixture(scope="module")
def learn():
    untar_data(Paths.MNIST)
    data = image_data_from_folder(Paths.MNIST, ds_tfms=(rand_pad(2, 28), []))
    learn = ConvLearner(data, tvm.resnet18, metrics=accuracy)
    learn.fit_one_cycle(1, 0.01)
    return learn

def test_accuracy(learn):
    assert accuracy(*learn.get_preds()) > 0.99

def test_image_data(learn):
    img,label = learn.data.train_ds[0]
    d = img.data
    assert abs(d.max()-1)<0.05
    assert abs(d.min())<0.05
    assert abs(d.mean()-0.2)<0.1
    assert abs(d.std()-0.3)<0.1

def test_1cycle_lrs(learn):
    lrs = learn.recorder.lrs
    assert lrs[0]<0.001
    assert lrs[-1]<0.0001
    assert np.max(lrs)==0.01

def test_1cycle_moms(learn):
    moms = learn.recorder.moms
    assert moms[0]==0.95
    assert abs(moms[-1]-0.95)<0.01
    assert np.min(moms)==0.85

