import pytest
from fastai import *
from fastai.vision import *

path = Paths.MNIST_TINY

@pytest.fixture(scope="module")
def data(request):
    untar_data(path)
    d = defaults.device
    defaults.device = torch.device('cpu')
    def _final(): defaults.device = d
    request.addfinalizer(_final)

@pytest.mark.skip(reason="pytorch bug")
def test_multi_iter_broken():
    data = image_data_from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    for i in range(5): x,y = next(iter(data.train_dl))

@pytest.mark.skip(reason="pytorch bug")
def test_multi_iter():
    data = image_data_from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    for i in range(5): x,y = data.train_dl.one_batch()

@pytest.mark.skip(reason="pytorch bug")
def test_normalize():
    data = image_data_from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    x,y = data.train_dl.one_batch()
    m,s = x.mean(),x.std()
    data.normalize()
    x,y = data.train_dl.one_batch()
    assert abs(x.mean()) < abs(m)
    assert abs(x.std()-1) < abs(m-1)

    with pytest.raises(Exception): data.normalize()
