import unittest

from fastai.layer_optimizer import LayerOptimizer


class Par(object):
    def __init__(self, x, grad=True):
        self.x = x
        self.requires_grad = grad
    def parameters(self): return [self]
def params_(*names): return [Par(nm) for nm in names]

class FakeOpt(object):
    def __init__(self, params): self.param_groups = params

class TestLayerOptimizer(unittest.TestCase):
    def test_init_atomic(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        self.check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])

    def test_init_list(self):
        lo = LayerOptimizer(
            FakeOpt,
            params_('A', 'B', 'C'),
            (1e-2, 2e-2, 3e-2),
            (9e-3, 8e-3, 7e-3),
        )
        self.check_optimizer_(
            lo.opt,
            [('A', 1e-2, 9e-3), ('B', 2e-2, 8e-3), ('C', 3e-2, 7e-3)],
        )

    def test_init_malformed_lr(self):
        with self.assertRaises(AssertionError):
            LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), (1e-2, 2e-2), 1e-4)

    def test_init_malformed_wd(self):
        with self.assertRaises(AssertionError):
            LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, (9e-3, 8e-3))

    def test_set_lrs_atomic(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        lo.set_lrs(1e-3)
        self.check_optimizer_(lo.opt, [(nm, 1e-3, 1e-4) for nm in 'ABC'])

    def test_set_lrs_list(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        lo.set_lrs([2e-2, 3e-2, 4e-2])
        self.check_optimizer_(
            lo.opt,
            [('A', 2e-2, 1e-4), ('B', 3e-2, 1e-4), ('C', 4e-2, 1e-4)],
        )

    def test_set_lrs_malformed(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        with self.assertRaises(AssertionError):
            lo.set_lrs([2e-2, 3e-2])
        self.check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])
        
    def test_set_wds_atomic(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        lo.set_wds(1e-5)
        self.check_optimizer_(lo.opt, [(nm, 1e-2, 1e-5) for nm in 'ABC'])

    def test_set_wds_list(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        lo.set_wds([9e-3, 8e-3, 7e-3])
        self.check_optimizer_(
            lo.opt,
            [('A', 1e-2, 9e-3), ('B', 1e-2, 8e-3), ('C', 1e-2, 7e-3)],
        )

    def test_set_wds_malformed(self):
        lo = LayerOptimizer(FakeOpt, params_('A', 'B', 'C'), 1e-2, 1e-4)
        with self.assertRaises(AssertionError):
            lo.set_wds([9e-3, 8e-3])
        self.check_optimizer_(lo.opt, [(nm, 1e-2, 1e-4) for nm in 'ABC'])

    def check_optimizer_(self, opt, expected):
        actual = opt.param_groups
        self.assertEqual(len(actual), len(expected))
        for (a, e) in zip(actual, expected): self.check_param_(a, *e)
        
    def check_param_(self, par, nm, lr, wd):
        self.assertEqual(par['params'][0].x, nm)
        self.assertEqual(par['lr'], lr)
        self.assertEqual(par['weight_decay'], wd)

if __name__ == '__main__':
    unittest.main()
