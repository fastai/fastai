import pytest
from fastai import *

def test_splitdata_datasets():
    c1,ratio,n = list('abc'),0.2,10

    sd = ItemList(range(n)).random_split_by_pct(ratio).label_const(0)
    assert len(sd.train)==(1-ratio)*n, 'Training set is right size'
    assert len(sd.valid)==ratio*n, 'Validation set is right size'
    assert set(list(sd.train.items)+list(sd.valid.items))==set(range(n)), 'All items covered'

def test_regression():
    df = pd.DataFrame({'x':range(100), 'y':np.random.rand(100)})
    data = ItemList.from_df(df, path='.', col=0).random_split_by_pct().label_from_df(cols=1).databunch()
    assert data.c==1
    assert isinstance(data.valid_ds, LabelList)