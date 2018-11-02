import pytest
from fastai import *
from fastai.vision import *

def test_valid_gets_train_classes():
    sds = (ImageFileList.from_folder(untar_data(URLs.MNIST_TINY))
           .label_from_folder().split_by_idx([0]).datasets(ImageClassificationDataset))
    assert np.array_equal(sds.train_ds.classes, sds.valid_ds.classes), 'train/valid classes same'

