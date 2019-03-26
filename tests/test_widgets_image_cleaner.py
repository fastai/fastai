import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.basics import *
from fastai.vision import *
from fastai.widgets import *

np.random.seed(42)

@pytest.fixture(scope="module")
def data():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=16, num_workers=2)
    return data

@pytest.mark.xfail(reason = "Expected Fail, lengths should be the same.")
def test_image_cleaner_index_length_mismatch(data):
    this_tests(ImageCleaner)
    with pytest.raises(AssertionError) as e:
        path = untar_data(URLs.MNIST_TINY)
        n = len(data.valid_ds)
        assert ImageCleaner(data.valid_ds, np.arange(n+2), path)

def test_image_cleaner_length_correct(data):
    this_tests(ImageCleaner)
    path = untar_data(URLs.MNIST_TINY)
    n = len(data.valid_ds)
    ImageCleaner(data.valid_ds, np.arange(n), path)

@pytest.mark.xfail(reason = "Expected Fail, Dataset should be passed instead.")
def test_image_cleaner_wrong_input_type(data):
    this_tests(ImageCleaner)
    path = untar_data(URLs.MNIST_TINY)
    n = len(data.valid_ds)
    ImageCleaner(data, np.arange(n), path)

def test_image_downloader_with_path():
    this_tests(ImageDownloader)
    ImageDownloader('.tmp/data')
