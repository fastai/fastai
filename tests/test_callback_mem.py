import pytest
from fastai.callbacks.mem import *
from utils.fakes import *
from utils.text import CaptureStdout

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="cuda enabled gpu is not available")

@cuda_required
def test_peak_mem_metric():
    learn = fake_learner()
    learn.callbacks.append(PeakMemMetric(learn))
    with CaptureStdout() as cs:
        learn.fit_one_cycle(3, max_lr=1e-2)
    for s in ['cpu', 'used', 'peak', 'gpu']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"
    # epochs 2-3 it shouldn't allocate more general or GPU RAM
    for s in ['0         0         0         0']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"
