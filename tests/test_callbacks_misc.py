import pytest
from fastai.callbacks.misc import *
from fastai.gen_doc.doctest import this_tests
from utils.fakes import *
from utils.text import CaptureStdout

def stop_after_n_batches_run_n_check(learn):
    with CaptureStdout() as cs:
        learn.fit_one_cycle(3, max_lr=1e-2)
    for s in ['train_loss', 'valid_loss']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"
    # test that epochs are stopped at epoch 0
    assert "\n0" in cs.out, "expecting epoch0"
    assert "\n1" not in cs.out, "epoch 1 shouldn't run"

    # test that only n batches were run
    assert len(learn.recorder.losses) == 2

def test_stop_after_n_batches():
    this_tests(StopAfterNBatches)

    # 1. global assignment
    defaults_extra_callbacks_bak = defaults.extra_callbacks
    defaults.extra_callbacks = [StopAfterNBatches(n_batches=2)]
    learn = fake_learner()
    stop_after_n_batches_run_n_check(learn)
    # restore
    defaults.extra_callbacks = defaults_extra_callbacks_bak

    # 2. dynamic assignment
    learn = fake_learner()
    learn.callbacks.append(StopAfterNBatches(n_batches=2))
    stop_after_n_batches_run_n_check(learn)
