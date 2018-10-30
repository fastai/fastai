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

def test_requires_grad():
    m = simple_cnn(b)
    assert requires_grad(m) == True

def test_requires_grad_set():
    m = simple_cnn(b)
    requires_grad(m,False)
    disj = False
    for p in m.parameters():
        disj = disj or p.requires_grad
    assert not disj
    requires_grad(m,True)
    conj = True
    for p in m.parameters():
        conj = conj and p.requires_grad
    assert conj

def test_apply_init():
    m = simple_cnn(b,bn=True)
    all2 = lambda m: nn.init.constant_(m.weight,0.2) if hasattr(m, 'weight') else m
    all7 = lambda m: nn.init.constant_(m,0.7)
    apply_leaf(m,all2)
    conv1_w = m[0][0].weight.clone()
    bn1_w = m[0][2].weight.clone()
    apply_init(m,all7)
    assert not conv1_w.equal(m[0][0].weight)
    assert bn1_w.equal(m[0][2].weight)
