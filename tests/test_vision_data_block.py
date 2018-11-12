import pytest
from fastai import *
from fastai.vision import *

def test_vision_datasets():
    sds = (ImageItemList.from_folder(untar_data(URLs.MNIST_TINY))
           .split_by_idx([0])
           .label_from_folder()
           .add_test_folder())
    assert np.array_equal(sds.train.classes, sds.valid.classes), 'train/valid classes same'
    assert len(sds.test)==20, "test_ds is correct size"

