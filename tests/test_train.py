# -*- coding: utf-8 -*-

import pytest
from fastai import *
from fastai.docs import *
from fastai.text import *



def test_lr_find_2_times_in_sequence():
    untar_data(IMDB_PATH)
    data_lm = text_data_from_csv(Path(IMDB_PATH), data_func=lm_data)
    learn = RNNLearner.language_model(data_lm)
    a = learn.lr_find()
    b = learn.lr_find()
    assert a == b