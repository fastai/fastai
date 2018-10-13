import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def data():
    path = Paths.MNIST_TINY
    untar_data(path)
    defaults.device = torch.device('cpu')
    data = image_data_from_folder(Paths.MNIST, ds_tfms=(rand_pad(2, 28), []))
    return data

def test_normalize(data):
    x,y = next(iter(data.train_dl))
    m,s = x.mean(),x.std()
    data.normalize()
    x,y = next(iter(data.train_dl))
    print(x.mean(),m,x.std(),s)
    assert abs(x.mean()) < abs(m)
    assert abs(x.std()-1) < abs(m-1)

    with pytest.raises(Exception): data.normalize()

