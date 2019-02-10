import pytest
from fastai.basics import *
from fastai.metrics import *
from utils.fakes import fake_learner
from utils.text import CaptureStdout

p1 = torch.Tensor([0,1,0,0,0]).expand(5,-1)
p2 = torch.Tensor([[0,0,0,0,0],[0,1,0,0,0]]).expand(5,2,-1).float()
t1 = torch.arange(5)
t2 = torch.Tensor([1,1,1,1,0]).expand(5,-1)

# Test data for multi-class single-label sequential models (like language models)
# batch size: 5, sequence length: 4, classes: 4
seq_targs = torch.arange(4).expand(5,-1)
seq_preds_perfect = torch.Tensor([
    [0.6, 0.3, 0.08, 0.02],
    [0.02, 0.6, 0.3, 0.08],
    [0.08, 0.02, 0.6, 0.3],
    [0.3, 0.08, 0.02, 0.6],
]).expand(5,4,-1)
seq_preds_wrong = torch.Tensor([
    [0.02, 0.6, 0.3, 0.08],
    [0.08, 0.02, 0.6, 0.3],
    [0.3, 0.08, 0.02, 0.6],
    [0.6, 0.3, 0.08, 0.02],
]).expand(5,4,-1)
seq_preds = torch.Tensor([
    [0.6, 0.3, 0.08, 0.02],
    [0.08, 0.3, 0.6, 0.02],
    [0.3, 0.02, 0.08, 0.6],
    [0.6, 0.3, 0.08, 0.02],
]).expand(5,4,-1)

@pytest.mark.parametrize("p, t, expect", [
    (p1, t1, 0.2),
    (torch.eye(5), t1, 1),
    (seq_preds_perfect, seq_targs, 1),
    (seq_preds_wrong, seq_targs, 0),
    (seq_preds, seq_targs, 0.25),
])
def test_accuracy(p, t, expect):
    assert np.isclose(accuracy(p, t).item(), expect)

@pytest.mark.parametrize("p, t, k, expect", [
    # should behave like `accuracy` for k = 1
    (p1, t1, 1, 0.2),
    (torch.eye(5), t1, 1, 1),
    (seq_preds_perfect, seq_targs, 1, 1),
    (seq_preds_wrong, seq_targs, 1, 0),
    (seq_preds, seq_targs, 1, 0.25),
    # should always return 1.0 for k = num_classes = 4
    (seq_preds_perfect, seq_targs, 4, 1),
    (seq_preds_wrong, seq_targs, 4, 1),
    (seq_preds, seq_targs, 4, 1),
    # perfect predictions should result in 1 for all k
    (seq_preds_perfect, seq_targs, 2, 1),
    (seq_preds_perfect, seq_targs, 3, 1),
    # totally wrong predictions should result in 0 for all k
    (seq_preds_wrong, seq_targs, 2, 0),
    (seq_preds_wrong, seq_targs, 3, 0),
    # all other cases
    (seq_preds, seq_targs, 2, 0.5),
    (seq_preds, seq_targs, 3, 0.75),
])
def test_top_k_accuracy(p, t, k, expect):
    assert np.isclose(top_k_accuracy(p, t, k).item(), expect)

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


### metric as a custom class
dummy_base_val = 9876
class DummyMetric(Callback):
    """ this dummy metric returns an epoch number power of a base """
    def __init__(self):
        super().__init__()
        self.name = "dummy"
        self.epoch = 0

    def on_epoch_begin(self, **kwargs):
        self.epoch += 1

    def on_epoch_end(self, **kwargs):
        self.metric = torch.tensor(dummy_base_val**self.epoch)

def test_custom_metric_class():
    learn = fake_learner(3,2)
    learn.metrics.append(DummyMetric())
    with CaptureStdout() as cs: learn.fit_one_cycle(2)
    # expecting column header 'dummy', and the metrics per class definition
    for s in ['dummy', f'{dummy_base_val}.00', f'{dummy_base_val**2}.00']:
        assert s in cs.out, f"{s} is in the output:\n{cs.out}"
