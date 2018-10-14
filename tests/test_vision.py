import pytest
from fastai import *
from fastai.vision import *

path = Paths.MNIST_TINY

@pytest.fixture(scope="module")
def data():
    untar_data(path)
    defaults.device = torch.device('cpu')

@pytest.mark.skip(reason="pytorch bug")
def test_multi_iter():
    data = image_data_from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    for i in range(10):
        print(i)
        x,y = next(iter(data.train_dl))

@pytest.mark.skip(reason="pytorch bug")
def test_normalize():
    data = image_data_from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    x,y = next(iter(data.train_dl))
    m,s = x.mean(),x.std()
    data.normalize()
    x,y = next(iter(data.train_dl))
    assert abs(x.mean()) < abs(m)
    assert abs(x.std()-1) < abs(m-1)

    with pytest.raises(Exception): data.normalize()
