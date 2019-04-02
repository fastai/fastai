import pytest
from fastai.callbacks.misc import *
from fastai.gen_doc.doctest import this_tests
from utils.fakes import *
from utils.text import CaptureStdout

def stop_after_n_batches_run_n_check(learn, bs, run_n_batches_exp):
    has_batches = len(learn.data.train_ds)//bs
    with CaptureStdout() as cs:
        learn.fit_one_cycle(3, max_lr=1e-2)
    for s in ['train_loss', 'valid_loss']:
        assert s in cs.out, f"expecting '{s}' in \n{cs.out}"

    # test that epochs are stopped at epoch 0
    assert "\n0" in cs.out, "expecting epoch0"
    assert "\n1" not in cs.out, "epoch 1 shouldn't run"

    # test that only run_n_batches_exp batches were run
    run_n_batches_got = len(learn.recorder.losses)
    assert run_n_batches_got == run_n_batches_exp, f"should have run only {run_n_batches_exp}, but got {run_n_batches_got}"

def test_stop_after_n_batches():
    this_tests(StopAfterNBatches)

    # this should normally give us 10 batches for train_ds
    train_length = 20
    bs = 2
    # but we only want to run 2
    run_n_batches = 2

    print()
    # 1. global assignment
    defaults_extra_callbacks_bak = defaults.extra_callbacks
    defaults.extra_callbacks = [StopAfterNBatches(n_batches=run_n_batches)]
    learn = fake_learner(train_length=train_length, batch_size=bs)
    stop_after_n_batches_run_n_check(learn, bs, run_n_batches)
    # restore
    defaults.extra_callbacks = defaults_extra_callbacks_bak

    # 2. dynamic assignment
    learn = fake_learner(train_length=train_length, batch_size=bs)
    learn.callbacks.append(StopAfterNBatches(n_batches=run_n_batches))
    stop_after_n_batches_run_n_check(learn, bs, run_n_batches)
