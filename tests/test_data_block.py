import pytest
from fastai import *

def test_splitdata_datasets():
    c1,ratio,n = list('abc'),0.2,10

    sd = ItemList(range(n)).random_split_by_pct(ratio).label_const(0)
    assert len(sd.train)==(1-ratio)*n, 'Training set is right size'
    assert len(sd.valid)==ratio*n, 'Validation set is right size'
    assert set(list(sd.train.items)+list(sd.valid.items))==set(range(n)), 'All items covered'
    sd = CategoryList(['a'], classes=c1).random_split_by_pct(0)
    assert np.array_equal(sd.train.classes, c1), 'train dataset classes correct'
    assert np.array_equal(sd.valid.classes, c1), 'validation dataset classes correct'

