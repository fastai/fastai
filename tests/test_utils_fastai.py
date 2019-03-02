import pytest, fastai
from fastai.gen_doc.doctest import this_tests

def test_has_version():
    this_tests('skip')
    assert fastai.__version__
