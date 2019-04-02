"""

"""

import pytest
from fastai.utils.mod_display import *
from utils.fakes import *
from utils.text import *
from fastai.gen_doc.doctest import this_tests


@pytest.fixture(scope="module")
def learn():
    learn = fake_learner(50,50)
    return learn

def test_progress_disabled_ctx(learn):
    this_tests(progress_disabled_ctx)
    #excpect output here
    with CaptureStdout() as cs: learn.fit(1)
    assert ('epoch' in cs.out) == True
    assert ('train_loss' in cs.out) == True
    assert ('valid_loss' in cs.out) == True
    assert ('\n0' in cs.out) == True #record of 0 epoch

    with CaptureStdout() as cs:
        with progress_disabled_ctx(learn):
            learn.fit(1)
    print(cs)
    assert ('epoch' not in cs.out) == True
    assert ('train_loss' not in cs.out) == True
    assert ('valid_loss' not in cs.out) == True
    assert ('\n0' not in cs.out) == True #record of 0 epoch


if __name__ == "__main__":
    test_progress_disabled_ctx(learn())