import pytest, torch, fastai

from fastai import torch_core

def test_show_install(capsys):
    fastai_version_check = f"fastai version : {fastai.__version__}"
    torch_version_check  = f"torch version  : {torch.__version__}"
    torch_core.show_install()
    captured = capsys.readouterr()
    #print(captured.out)
    assert fastai_version_check in captured.out
    assert torch_version_check  in captured.out
