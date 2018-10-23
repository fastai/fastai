import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path(request):
    path = untar_data(URLs.MNIST_TINY)
    d = defaults.device
    defaults.device = torch.device('cpu')
    def _final(): defaults.device = d
    request.addfinalizer(_final)
    return path

def test_path_can_be_str_type(path):
    assert ImageDataBunch.from_csv(str(path))

def test_multi_iter_broken(path):
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    for i in range(2): x,y = next(iter(data.train_dl))

def test_multi_iter(path):
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    for i in range(2): x,y = data.train_dl.one_batch()

def test_clean_tear_down(path):
    docstr = "test DataLoader iter doesn't get stuck"
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()

def test_normalize(path):
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    x,y = data.valid_dl.one_batch()
    m,s = x.mean(),x.std()
    data.normalize()
    x,y = data.valid_dl.one_batch()
    assert abs(x.mean()) < abs(m)
    assert abs(x.std()-1) < abs(m-1)

    with pytest.raises(Exception): data.normalize()
