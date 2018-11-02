import pytest
from fastai import *

def test_splitdata_datasets():
    c1,ratio,n = list('abc'),0.2,10
    c2 = c1+['d']

    sd = InputList(range(n)).label_from_func(lambda x:x).random_split_by_pct(ratio)
    assert len(sd.train)==(1-ratio)*n, 'Training set is right size'
    assert len(sd.valid)==ratio*n, 'Validation set is right size'
    assert set(list(sd.train.files)+list(sd.valid.files))==set(range(n)), 'All items covered'

    sds = sd.datasets(LabelXYDataset, classes=c1)
    assert np.array_equal(sds.train_ds.classes, c1), 'train dataset classes correct'
    assert np.array_equal(sds.valid_ds.classes, c1), 'validation dataset classes correct'

