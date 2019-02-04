import pytest
from utils.fakes import *
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
    learn = fake_learner()
    x,y = next(iter(learn.data.train_dl))
    res1 = learn.model(x)
    learn.model = model2half(learn.model)
    res2 = learn.model(x.half())
    assert (res2.float() - res1).abs().sum() < 0.01

def test_to_half():
    t1,t2 = torch.ones([1]).long(),torch.ones([1])
    half = to_half([t1,t2])
    assert isinstance(half[0],torch.LongTensor)
    assert isinstance(half[1],torch.HalfTensor)
    
def test_batch_to_half():
    t1,t2 = torch.ones([1]),torch.ones([1])
    half = batch_to_half([t1,t2])
    assert isinstance(half[0],torch.HalfTensor)
    assert isinstance(half[1],torch.FloatTensor)    
