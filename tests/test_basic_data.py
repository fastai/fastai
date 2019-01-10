import pytest
from utils.fakes import *
from io import StringIO
from contextlib import redirect_stdout
import sys

## run: pytest tests/test_basic_data.py -s


# filename: test_basic_data.py

## Test Cases

    ## TO DO: Function intercept_args

    ## TO DO: Class DeviceDataLoader

## Class DataBunch

def test_DataBunch_Create():
    x_train,y_train =  fake_basedata(n_in=3, batch_size=6),fake_basedata(n_in=3, batch_size=6)
    x_valid,y_valid =  fake_basedata(n_in=3, batch_size=3),fake_basedata(n_in=3, batch_size=3)
    bs=5
    train_ds,valid_ds = TensorDataset(x_train, y_train),TensorDataset(x_valid, y_valid)
    data = DataBunch.create(train_ds, valid_ds, bs=bs)
   
    assert 3 == len(data.train_dl)
    assert 18 == len(data.train_ds)
    assert 2 == len(data.valid_dl)
    assert 9 == len(data.valid_ds)

## TO DO (?)ideally, call one_batch with type dataloader    
def test_DataBunch_onebatch():
    data = fake_data(n_in=4, n_out=5, batch_size=6)
    x,y = data.one_batch()
    assert 4 == x[0].shape[0] 
    assert 6 == x.shape[0] 
    assert 6 == y.shape[0] 

    
def test_DataBunch_oneitem():
    data = fake_data()
    x,y = data.one_item(item=1)
    assert 1 == x.shape[0]
    assert 1 == y.shape[0]
    
    
def test_DataBunch_show_batch(capsys):
     data = fake_data()
     data.show_batch()
     captured = capsys.readouterr()
     match = re.findall(r'tensor', captured.out)
     assert match

## TO DO: check over file created ?    
##def test_DataBunch_export():     
##     data = fake_data()
##     data.export()


