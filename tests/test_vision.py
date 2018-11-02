import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def test_path_can_be_str_type(path):
    assert ImageDataBunch.from_csv(str(path))