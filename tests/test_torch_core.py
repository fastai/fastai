import pytest, torch, fastai
from fastai.torch_core import *
from math import isclose

a=[1,2,3]
exp=torch.tensor(a)
a3b3b3 =torch.ones([1,3,3,3])

class ConvBNNet(torch.nn.Module):
    def __init__(self, D_in):
        super(ConvBNNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(D_in, D_in*2, 2)
        self.bn = torch.nn.BatchNorm2d(D_in*2)

    def forward(self, x):
        h_relu = self.conv1(x).clamp(min=0)
        return self.bn(h_relu)

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

def test_model2half():
    m = ConvBNNet(3)
    half_m = model2half(m)
    assert isinstance(half_m.conv1.weight, torch.HalfTensor)
    assert isinstance(half_m.bn.weight, torch.FloatTensor)
    result = half_m.cuda().forward(a3b3b3.cuda().half()).sum()
    assert isclose(result,0,abs_tol=1e-3)
