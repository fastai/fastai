import pytest, torch, re, fastai
from fastai.gen_doc.doctest import this_tests
from fastai.torch_core import *
from fastai.utils.show_install import *
from fastai.utils.check_perf import *
from PIL import Image

def test_show_install(capsys):
    this_tests(show_install)
    show_install()
    captured = capsys.readouterr()
    #print(captured.out)
    match = re.findall(rf'fastai\s+: {fastai.__version__}', captured.out)
    assert match
    match = re.findall(rf'torch\s+: {re.escape(torch.__version__)}', captured.out)
    assert match

def test_check_perf(capsys):
    this_tests(check_perf)
    check_perf()
    captured = capsys.readouterr()
    #print(captured.out)
    #match = re.findall(rf'Running Pillow.*?{Image.PILLOW_VERSION}', captured.out)
    #assert match
