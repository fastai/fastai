import pytest, torch, fastai
from fastai.torch_core import *
from fastai.layers import *

a=[1,2,3]
exp=torch.tensor(a)
b=[3,6,6]

def test_tensor_with_list():
    r = tensor(a)
    assert torch.all(r==exp)

def test_tensor_with_ndarray():
    b=np.array(a)
    r = tensor(b)
    assert np_address(r.numpy()) == np_address(b)
    assert torch.all(r==exp)

def test_tensor_with_tensor():
    c=torch.tensor(a)
    r = tensor(c)
    assert r.data_ptr()==c.data_ptr()
    assert torch.all(r==exp)

def test_in_channels():
    m = simple_cnn(b)
    assert in_channels(m) == 3

def test_in_channels_no_weights():
    with pytest.raises(Exception) as e_info:
        in_channels(nn.Sequential())
    assert e_info.value.args[0] == 'No weight layer'
