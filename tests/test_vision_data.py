import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def mnist_tiny_sanity_test(data):
    assert data.c == 2
    assert set(map(str, set(data.classes))) == {'3', '7'}
    assert set(data.train_ds.y) == set(data.valid_ds.y) == {0, 1}

def test_from_folder(path):
    for valid_pct in [None, 0.9]:
        data = ImageDataBunch.from_folder(path, test='test', valid_pct=valid_pct)
        mnist_tiny_sanity_test(data)

def test_from_name_re(path):
    fnames = get_files(path/'train', recurse=True)
    pat = r'\/([^/]+)\/\d+.png$'
    data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=(rand_pad(2, 28), []))
    mnist_tiny_sanity_test(data)

def test_from_csv_and_from_df(path):
    for func in ['from_csv', 'from_df']:
        files = []
        if func is 'from_df': data = ImageDataBunch.from_df(path, df=pd.read_csv(path/'labels.csv'), size=28)
        else: data = ImageDataBunch.from_csv(path, size=28)
        mnist_tiny_sanity_test(data)