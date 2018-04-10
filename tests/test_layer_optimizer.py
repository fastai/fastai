import pytest

from fastai.layer_optimizer import LayerOptimizer


class Par(object):
    def __init__(self, x, grad=True):
        self.x = x
        self.requires_grad = grad
    def parameters(self): return [self]

class FakeOpt(object):
    def __init__(self, params): self.param_groups = params

def params_(*names): return [Par(nm) for nm in names]

def check_optimizer_(opt, expected):
    actual = opt.param_groups
    assert len(actual) == len(expected)
    for (a, e) in zip(actual, expected): check_param_(a, *e)
    
def check_param_(par, nm, lr, wd):
    assert par['params'][0].x == nm
    assert par['lr'] == lr
    assert par['weight_decay'] == wd


def test_construction_with_singleton_lr_and_wd():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])

def test_construction_with_lists_of_lrs_and_wds():
    lo = LayerOptimizer(
        FakeOpt,
        params_('A', 'B', 'C'),
        (1e-2, 2e-2, 3e-2),
        (9e-3, 8e-3, 7e-3),
    )
    check_optimizer_(
        lo.opt,
        [('A', 1e-2, 9e-3), ('B', 2e-2, 8e-3), ('C', 3e-2, 7e-3)],
    )

def test_construction_with_too_few_lrs():
    with pytest.raises(AssertionError):
        LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), (1e-2, 2e-2), 1e-4)

def test_construction_with_too_few_wds():
    with pytest.raises(AssertionError):
        LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, (9e-3, 8e-3))

def test_set_lrs_with_single_value():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    lo.set_lrs(1e-3)
    check_optimizer_(lo.opt, [(nm, 1e-3, 1e-4) for nm in 'ABC'])

def test_set_lrs_with_list_of_values():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    lo.set_lrs([2e-2, 3e-2, 4e-2])
    check_optimizer_(
        lo.opt,
        [('A', 2e-2, 1e-4), ('B', 3e-2, 1e-4), ('C', 4e-2, 1e-4)],
    )

def test_set_lrs_with_too_few_values():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    with pytest.raises(AssertionError):
        lo.set_lrs([2e-2, 3e-2])
    # Also make sure the optimizer didn't change.
    check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])
    
def test_set_wds_with_single_value():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    lo.set_wds(1e-5)
    check_optimizer_(lo.opt, [(nm, 1e-2, 1e-5) for nm in 'ABC'])

def test_set_wds_with_list_of_values():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    lo.set_wds([9e-3, 8e-3, 7e-3])
    check_optimizer_(
        lo.opt,
        [('A', 1e-2, 9e-3), ('B', 1e-2, 8e-3), ('C', 1e-2, 7e-3)],
    )

def test_set_wds_with_too_few_values():
    lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
    with pytest.raises(AssertionError):
        lo.set_wds([9e-3, 8e-3])
    # Also make sure the optimizer didn't change.
    check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])
