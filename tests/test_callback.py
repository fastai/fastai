import pytest, fastai
from fastai.vision import *
from io import StringIO
from contextlib import redirect_stdout
from utils.fakes import fake_data
from utils.text import apply_print_resets

n_in, n_out = 3, 2
@pytest.fixture(scope="module")
def data(): return fake_data(n_in=n_in, n_out=n_out)

@pytest.fixture(scope="module")
def model(): return nn.Linear(n_in, n_out)

col_a = 5678722929
col_b = 1237892223
class DummyCallback(LearnerCallback):
    _order=-20
    def __init__(self, learn):
        super().__init__(learn)
        self.dummy = 0

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['col_a', 'col_b'])

    def on_epoch_end(self, **kwargs):
        # add dummy metrics
        self.learn.recorder.add_metrics([col_a, col_b])

def check_dummy_metric(buffer):
    out = apply_print_resets(buffer.getvalue())
    for s in ['col_a', col_a, 'col_b', col_b]:
        assert str(s) in out, f"{s} is in the output:\n{out}"

def test_callbacks_learner(data, model):

    # single callback in learner constructor
    learn = Learner(data, model, metrics=accuracy, callback_fns=DummyCallback)
    buffer = StringIO()
    with redirect_stdout(buffer): learn.fit_one_cycle(2)
    check_dummy_metric(buffer)

    # list of callbacks in learner constructor
    learn = Learner(data, model, metrics=accuracy, callback_fns=[DummyCallback])
    buffer = StringIO()
    with redirect_stdout(buffer): learn.fit_one_cycle(2)
    check_dummy_metric(buffer)

    # single callback append
    learn = Learner(data, model, metrics=accuracy)
    learn.callbacks.append(DummyCallback(learn))
    buffer = StringIO()
    with redirect_stdout(buffer): learn.fit_one_cycle(2)
    check_dummy_metric(buffer)

    # list of callbacks append: python's append, so append([x]) will not do the right
    # thing, so it's expected to fail
    learn = Learner(data, model, metrics=[accuracy])
    learn.callbacks.append([DummyCallback(learn)])
    error = ''
    buffer = StringIO()
    try:
        with redirect_stdout(buffer): learn.fit_one_cycle(2)
    except Exception as e:
        error = str(e)
    error_pat = "'list' object has no attribute 'on_train_begin'"
    assert error_pat in error, f"{error_pat} is in the exception:\n{error}"

def test_callbacks_fit(data, model):
    learn = Learner(data, model, metrics=accuracy)

    for func in ['fit', 'fit_one_cycle']:
        fit_func = getattr(learn, func)

        # single callback
        buffer = StringIO()
        with redirect_stdout(buffer): fit_func(2, callbacks=DummyCallback(learn))
        check_dummy_metric(buffer)

        # list of callbacks
        buffer = StringIO()
        with redirect_stdout(buffer): fit_func(2, callbacks=[DummyCallback(learn)])
        check_dummy_metric(buffer)
