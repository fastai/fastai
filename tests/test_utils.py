import pytest, torch, re, fastai
from fastai.torch_core import *
from fastai.utils.collect_env import *

def test_show_install(capsys):
    show_install()
    captured = capsys.readouterr()
    #print(captured.out)
    match = re.findall(rf'fastai version\s+: {fastai.__version__}', captured.out)
    assert match
    match = re.findall(rf'torch version\s+: {torch.__version__}',   captured.out)
    assert match
