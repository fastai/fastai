import pytest
from utils.fakes import *
import sys
from fastai.gen_doc.doctest import this_tests

## run: pytest tests/test_basic_data.py -s


# filename: test_basic_data.py

## Test Cases

    ## TO DO: Function intercept_args

    ## TO DO: Class DeviceDataLoader

## Class DataBunch

def test_DataBunch_Create():
    x_train,y_train = fake_basedata(n_in=3, batch_size=6),fake_basedata(n_in=3, batch_size=6)
    x_valid,y_valid = fake_basedata(n_in=3, batch_size=3),fake_basedata(n_in=3, batch_size=3)
    bs=5
    train_ds,valid_ds = TensorDataset(x_train, y_train),TensorDataset(x_valid, y_valid)
    data = DataBunch.create(train_ds, valid_ds, bs=bs)
    this_tests(data.create)

    assert 4 == len(data.dls)
    assert 3 == len(data.train_dl)
    assert 18 == len(data.train_ds)
    assert 2 == len(data.valid_dl)
    assert 9 == len(data.valid_ds)

def test_DataBunch_no_valid_dl():
    x_train,y_train = fake_basedata(n_in=3, batch_size=6),fake_basedata(n_in=3, batch_size=6)
    bs=5
    train_ds = TensorDataset(x_train, y_train)
    data = DataBunch.create(train_ds, None, bs=bs)
    this_tests(data.create)
    data.valid_dl = None

    assert 3 == len(data.dls)
    assert 3 == len(data.train_dl)
    assert 18 == len(data.train_ds)
    assert None == data.valid_dl

## TO DO (?)ideally, call one_batch with type dataloader
def test_DataBunch_onebatch():
    data = fake_data(n_in=4, n_out=5, batch_size=6)
    this_tests(data.one_batch)
    x,y = data.one_batch()
    assert 4 == x[0].shape[0]
    assert 6 == x.shape[0]
    assert 6 == y.shape[0]


def test_DataBunch_oneitem():
    data = fake_data()
    this_tests(data.one_item)
    x,y = data.one_item(item=1)
    assert 1 == x.shape[0]
    assert 1 == y.shape[0]


def test_DataBunch_show_batch(capsys):
    data = fake_data()
    this_tests(data.show_batch)
    data.show_batch()
    captured = capsys.readouterr()
    match = re.findall(r'tensor', captured.out)
    assert match

def test_DataBunch_save_load():
    save_name = 'data_save.pkl'
    this_tests(DataBunch.save, load_data)

    data = fake_data(n_in=4, n_out=5, batch_size=6)
    data.save(save_name)
    loaded_data = load_data(data.path, save_name, bs=6)
    this_tests(loaded_data.one_batch)
    x,y = loaded_data.one_batch()
    assert 4 == x[0].shape[0]
    assert 6 == x.shape[0]
    assert 6 == y.shape[0]

    # save/load using buffer
    output_buffer = io.BytesIO()
    data.save(output_buffer)
    input_buffer = io.BytesIO(output_buffer.getvalue())
    loaded_data = load_data(data.path, input_buffer, bs=6)
    this_tests(loaded_data.one_batch)
    x,y = loaded_data.one_batch()
    assert 4 == x[0].shape[0]
    assert 6 == x.shape[0]
    assert 6 == y.shape[0]
    os.remove(save_name)

def test_DeviceDataLoader_getitem():
    this_tests('na')
    class DictDataset(Dataset):
        def __getitem__(self, idx):
            return {"a":np.ones((3,)),"b":np.zeros((2,))}
        def __len__(self):
            return 10

    ds = DictDataset()
    next(iter(DeviceDataLoader.create(ds)))
