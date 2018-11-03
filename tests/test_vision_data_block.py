import pytest
from fastai import *
from fastai.vision import *

def test_vision_datasets():
    sds = (ImageFileList.from_folder(untar_data(URLs.MNIST_TINY))
           .label_from_folder().split_by_idx([0]).add_test_folder().datasets(ImageClassificationDataset))
    assert np.array_equal(sds.train_ds.classes, sds.valid_ds.classes), 'train/valid classes same'
    assert len(sds.test_ds)==20, "test_ds is correct size"

