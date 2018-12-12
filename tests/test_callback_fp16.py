import pytest, torch, fastai
from fastai.torch_core import *
from fastai.layers import *
from math import isclose
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="cuda enabled gpu is not available")
a3b3b3 =torch.ones([1,3,3,3])

def test_model2half():
    m = simple_cnn([3,6,6],bn=True)
    m = model2half(m)
    conv1 = m[0][0]
    bn = m[0][2]
    assert isinstance(conv1.weight, torch.HalfTensor)
    assert isinstance(bn.weight, torch.FloatTensor)

@cuda_required
def test_model2half_forward():
    m = simple_cnn([3,6,6],bn=True)
    m = model2half(m)
    bn_result = m[0].cuda().forward(a3b3b3.cuda().half()).sum()
    assert isclose(bn_result,0,abs_tol=3e-2)

def test_to_half():
    t1,t2 = torch.ones([1]),torch.ones([1])
    half = to_half([t1,t2])
    assert isinstance(half[0],torch.HalfTensor)
    assert isinstance(half[1],torch.FloatTensor)
