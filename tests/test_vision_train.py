import pytest
from fastai import *
from fastai.vision import *

#@pytest.mark.slow
@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
    data.normalize()
    learn = Learner(data, simple_cnn((3,16,16,16,2), bn=True), metrics=accuracy)
    learn.fit_one_cycle(3)
    return learn

def test_accuracy(learn):
    assert accuracy(*learn.get_preds()) > 0.9

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
    assert np.max(lrs)==3e-3

def test_1cycle_moms(learn):
    moms = learn.recorder.moms
    assert moms[0]==0.95
    assert abs(moms[-1]-0.95)<0.01
    assert np.min(moms)==0.85

