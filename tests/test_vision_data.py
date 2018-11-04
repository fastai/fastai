import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def test_from_folder(path):
    for valid_pct in [None, 0.9]:
        data = ImageDataBunch.from_folder(path, test='test', valid_pct=valid_pct)
        assert len(data.classes) == 2
        assert set(data.classes) == set(['3', '7'])

def test_from_name_re(path):
    fnames = get_files(path/'train', recurse=True)
    pat = r'\/([^/]+)\/\d+.png$'
    data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=(rand_pad(2, 28), []))
    assert data.c == 2
    assert {'3', '7'} == set(data.classes)
    assert {0, 1} == set(data.train_ds.y)
    assert {0, 1} == set(data.valid_ds.y)

def test_from_df_test_dataset(path):
    "Check that test dataset is created with from_df." 
    #Actual contents do not matter here.
    df = pd.DataFrame({'fn': ['a.jpg', 'b.jpg', 'c.jpg'],
                       'lbl': [0, 1, 2]})
    data = ImageDataBunch.from_df('.', df, path, test='test')
    #If the test set is not registered, this will raise an assertion error
    data.test_ds
