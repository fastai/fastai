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

def test_multi():
    path = untar_data(URLs.PLANET_TINY)
    data = (ImageItemList.from_csv(path, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct().label_from_df(sep=' ').databunch())
    x,y = data.valid_ds[0]
    assert x.shape[0]==3
    assert data.c==len(y.data)==14
    assert len(str(y))>2

