import pytest
from fastai.callbacks.mem import *
from fastai.gen_doc.doctest import this_tests
from utils.fakes import *
from utils.text import CaptureStdout

@pytest.mark.skip("occassional random failures")
@pytest.mark.cuda
def test_peak_mem_metric():
    learn = fake_learner()
    learn.callbacks.append(PeakMemMetric(learn))
    this_tests(PeakMemMetric)
    with CaptureStdout() as cs:
        learn.fit_one_cycle(3, max_lr=1e-2)
    for s in ['cpu', 'used', 'peak', 'gpu']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"
    # XXX: needs a better test to assert some numbers here (at least >0)
    # epochs 2-3 it shouldn't allocate more general or GPU RAM
    for s in ['0         0         0         0']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"
