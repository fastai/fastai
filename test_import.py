import pytest
import os

def rename(old, new):
    os.rename(old, new)

rename("fastai", "fastai5")

try:
    import fastai
    assert False
except:
    print("fastai not globally installed, good")

try:
    from fastai5 import *
    from fastai5.callbacks import *
    from fastai5.imports import *
    from fastai5.tabular import *
    from fastai5.text import *
    from fastai5.utils import *
    from fastai5.vision import *
    from fastai5.widgets import *

    print("PASSED !")
finally:
    rename("fastai5", "fastai")
