import pytest
from fastai import *
from fastai.metrics import *

p1 = torch.Tensor([0,1,0,0,0]).expand(5,-1)
p2 = torch.Tensor([[0,0,0,0,0],[0,1,0,0,0]]).expand(5,2,-1).float()
t1 = torch.arange(5)
t2 = torch.Tensor([1,1,1,1,0]).expand(5,-1)

@pytest.mark.parametrize("p, t, expect", [
    (p1, t1, 0.2),
    (torch.eye(5), t1, 1),
])
def test_accuracy(p, t, expect):
    assert np.isclose(accuracy(p, t).item(), expect)

@pytest.mark.parametrize("p, t, expect", [
    (p1, t1, 0.8),
    (torch.eye(5), t1, 0),
])
def test_error_rate(p, t, expect):
    assert np.isclose(error_rate(p, t).item(), expect)

def test_exp_rmspe():
    assert np.isclose(exp_rmspe(torch.ones(1,5), torch.ones(5)).item(), 0)

def test_exp_rmspe_num_of_ele():
    with pytest.raises(AssertionError):
        exp_rmspe(p1, t1.float())

def test_accuracy_thresh():
    assert np.isclose(accuracy_thresh(torch.linspace(0,1,5), torch.ones(5)), 0.8)

@pytest.mark.parametrize("p, t, expect", [
    (p2, t2, 0.4),
    (torch.zeros(5,2,5), torch.eye(5,5), 1/3),
    (torch.zeros(5,2,5), torch.zeros(5,5), 0),
])
def test_dice(p, t, expect):
    assert np.isclose(dice(p, t.long()).item(), expect)

@pytest.mark.parametrize("p, t, expect", [
    (p2, t2, 0.238095),
    (p2, torch.eye(5,5), 0.1),
    (p2, torch.zeros(5,5), 0),
])
def test_dice_iou(p, t, expect):
    assert np.isclose(dice(p, t.long(), iou=True).item(), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.ones(1,10), torch.ones(1,10), 1),
    (torch.zeros(1,10), torch.zeros(1,10), 0),
])
def test_fbeta(p, t, expect):
    assert np.isclose(fbeta(p, t).item(), expect)

