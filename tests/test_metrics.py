import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.basics import *
from fastai.metrics import *
from utils.fakes import fake_learner
from utils.text import CaptureStdout

p1 = torch.Tensor([0,1,0,0,0]).expand(5,-1)
p2 = torch.Tensor([[0,0,0,0,0],[0,1,0,0,0]]).expand(5,2,-1).float()
p3 = torch.Tensor([[0,0,0,0,0],[0,0,0,0,0]]).expand(5,2,-1).float()
t1 = torch.arange(5)
t2 = torch.Tensor([1,1,1,1,0]).expand(5,-1)
t3 = torch.Tensor([0,0,0,0,0]).expand(5,-1)

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
    this_tests(accuracy)
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
    this_tests(top_k_accuracy)
    assert np.isclose(top_k_accuracy(p, t, k).item(), expect)

@pytest.mark.parametrize("p, t, expect, atol", [
    (torch.randn((128, 2, 224, 224)),  torch.randint(0, 2, (128, 1, 224, 224)), 1/2, 1e-3),
    (torch.randn((128, 8, 224, 224)),  torch.randint(0, 8, (128, 1, 224, 224)), 1/8, 1e-3),
    (torch.randn((128, 16, 224, 224)),  torch.randint(0, 16, (128, 1, 224, 224)), 1/16, 1e-3),
])
def test_foreground_acc(p, t, expect, atol):
    this_tests(foreground_acc)
    assert np.isclose(partial(foreground_acc, void_code=0)(p, t).item(), expect, atol=atol)

@pytest.mark.parametrize("p, t, expect", [
    (p1, t1, 0.8),
    (torch.eye(5), t1, 0),
])
def test_error_rate(p, t, expect):
    this_tests(error_rate)
    assert np.isclose(error_rate(p, t).item(), expect)

def test_exp_rmspe():
    this_tests(exp_rmspe)
    assert np.isclose(exp_rmspe(torch.ones(1,5), torch.ones(5)).item(), 0)

def test_exp_rmspe_num_of_ele():
    this_tests(exp_rmspe)
    with pytest.raises(AssertionError):
        exp_rmspe(p1, t1.float())

def test_accuracy_thresh():
    this_tests(accuracy_thresh)
    assert np.isclose(accuracy_thresh(torch.linspace(0,1,5), torch.ones(5)), 0.8)

@pytest.mark.parametrize("p, t, expect", [
    (p2, t2, 8/9), #0.4
    (torch.zeros(5,2,5), torch.eye(5,5), 1/3),
    (torch.zeros(5,2,5), torch.zeros(5,5), 0),
    (tensor([[[[1., 1.],
               [1., 1.]],
              [[0., 0.],
               [0., 0.]]],
             [[[1., 1.],
               [1., 1.]],
              [[1., 0.],
               [0., 0.]]],
             [[[1., 1.],
               [1., 1.]],
              [[0., 0.],
               [1., 0.]]]]),
     tensor([[[[0, 0],
               [0, 0]]],
             [[[0, 0],
               [0, 1]]],
             [[[0, 0],
               [1, 0]]]]),
     2/3)
])
def test_dice(p, t, expect):
    this_tests(dice)
    assert np.isclose(dice(p, t.long()).item(), expect)

@pytest.mark.parametrize("p, t, expect, atol", [
    (p2, t2, 0.8, 0.),
    (p3, t3, 0.0, 0.),
    (p2, torch.eye(5,5), 0.200, 1e-3),
    (p2, torch.zeros(5,5), 0, 0.),
])
def test_dice_iou(p, t, expect, atol):
    this_tests(dice)
    assert np.isclose(dice(p, t.long(), iou=True).item(), expect, atol=atol)

@pytest.mark.parametrize("p, t, expect", [
    (torch.ones(1,10), torch.ones(1,10), 1),
    (torch.zeros(1,10), torch.zeros(1,10), 0),
])
def test_fbeta(p, t, expect):
    this_tests(fbeta)
    assert np.isclose(fbeta(p, t).item(), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.arange(-10, 10).float(), torch.arange(-9, 11).float(), 1),
    (torch.arange(-10, 10).float(), torch.arange(-11, 9).float(), 1),
    (torch.arange(-10, 10).float(), torch.arange(-10, 10).float(), 0)
])
def test_mae(p, t, expect):
    this_tests(mean_absolute_error)
    assert np.isclose(mean_absolute_error(p, t), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.arange(-10, 10).float(), torch.arange(-8, 12).float(), 4),
    (torch.arange(-10, 10).float(), torch.arange(-12, 8).float(), 4),
    (torch.arange(-10, 10).float(), torch.arange(-10, 10).float(), 0)
])
def test_mse(p, t, expect):
    this_tests(mean_squared_error)
    assert np.isclose(mean_squared_error(p, t), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.arange(-10, 10).float(), torch.arange(-8, 12).float(), 2),
    (torch.arange(-10, 10).float(), torch.arange(-12, 8).float(), 2),
    (torch.arange(-10, 10).float(), torch.arange(-10, 10).float(), 0)
])
def test_rmse(p, t, expect):
    this_tests(root_mean_squared_error)
    assert np.isclose(root_mean_squared_error(p, t), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.exp(torch.arange(-10, 10).float())-1,
     torch.exp(torch.arange(-8,  12).float())-1, 4),
    (torch.exp(torch.arange(-10, 10).float())-1,
     torch.exp(torch.arange(-12,  8).float())-1, 4),
])
def test_msle(p, t, expect):
    this_tests(mean_squared_logarithmic_error)
    assert np.isclose(mean_squared_logarithmic_error(p, t), expect, rtol=1.e-4)

@pytest.mark.parametrize("p, t, expect", [
    (torch.arange(-5, 5).float(), torch.arange(-10,  0).float(), 1.),
    (torch.zeros(10).float(), torch.arange(-5, 5).float(), 0.),
    (torch.arange(-5, 5).float(), torch.zeros(10).float(), -float("inf")),
    (p1/2., p1, 0.75),
    (p1, t2, -0.5),
])
def test_explained_variance(p, t, expect):
    this_tests(explained_variance)
    assert np.isclose(explained_variance(p, t), expect)

@pytest.mark.parametrize("p, t, expect", [
    (torch.arange(-5, 5).float(), torch.arange(-5,  5).float(), 1),
    (torch.zeros(10).float(), torch.arange(-5, 5).float(), -0.0303),
    (torch.arange(-5, 5).float(), torch.zeros(10).float(), -float("inf")),
    (p1/2., p1, 0.6875),
    (p1, t2, -2.75),
])
def test_r2_score(p, t, expect):
    this_tests(r2_score)
    assert np.isclose(r2_score(p, t), expect, atol=1e-2)

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

    def on_epoch_end(self, last_metrics, **kwargs):
        return {'last_metrics': last_metrics + [torch.tensor(dummy_base_val**self.epoch)]}

def test_custom_metric_class():
    this_tests('na')
    learn = fake_learner(3,2)
    learn.metrics.append(DummyMetric())
    with CaptureStdout() as cs: learn.fit_one_cycle(2)
    # expecting column header 'dummy', and the metrics per class definition
    for s in ['dummy', f'{dummy_base_val}.00', f'{dummy_base_val**2}.00']:
        assert s in cs.out, f"{s} is in the output:\n{cs.out}"

def test_average_metric_naming():
    this_tests(AverageMetric)
    top2_accuracy = partial(top_k_accuracy, k=2)
    top3_accuracy = partial(top_k_accuracy, k=3)
    top4_accuracy = partial(top_k_accuracy, k=4)
    # give top2_accuracy and top4_accuracy a custom name
    top2_accuracy.__name__ = "top2_accuracy"
    top4_accuracy.__name__ = "top4_accuracy"
    # prewrap top4_accuracy
    top4_accuracy = AverageMetric(top4_accuracy)
    learn = fake_learner()
    learn.metrics = [accuracy, top2_accuracy, top3_accuracy, top4_accuracy]
    learn.fit(1)
    assert learn.recorder.names[3:7] == ["accuracy", "top2_accuracy", "top_k_accuracy", "top4_accuracy"]