"""Tests for `fastai.vision` package."""

import pytest
from fastai import *
from fastai.vision import *
from fastai.docs import *

class TestVision():
    def test_mnist_end_to_end(self):
        untar_data(MNIST_PATH)
        data = image_data_from_folder(MNIST_PATH, ds_tfms=(rand_pad(2, 28), []))
        learn = ConvLearner(data, tvm.resnet18, metrics=accuracy)
        learn.fit(1)
        assert accuracy(*learn.get_preds()) > 0.98

