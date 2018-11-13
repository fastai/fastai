import pytest
from fastai import *
from fastai.vision import *
from fastai.widgets import *

np.random.seed(42)
@pytest.mark.xfail(reason = "Expected Fail")
def test_image_cleaner_length_long():
    with pytest.raises(AssertionError) as e_info:
        path = untar_data(URLs.MNIST_TINY)
        data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
        n = len(data.valid_ds)
    assert  ImageCleaner(data.valid_ds, np.arange(n+1))
    
@pytest.mark.xfail(reason = "Expected Fail")                   
def test_image_cleaner_length_short():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
    n = len(data.valid_ds)
    assert ImageCleaner(data.valid_ds, np.arange(n+1))
    
def test_image_cleaner_length_correct():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
    n = len(data.valid_ds)
 
