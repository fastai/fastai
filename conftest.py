import pytest
import re

def pytest_itemcollected(item):
    #import pdb; pdb.set_trace()
    par = item.parent.obj
    node = item.obj
    pref = par.__doc__.strip() if par.__doc__ else par.__class__.__name__
    pref = re.sub(r'^Test', '', pref)
    if pref=='module': pref = re.sub(r'^test_', '', par.__name__)
    suf = node.__doc__.strip() if node.__doc__ else node.__name__
    suf = re.sub(r'^test_', '', suf)
    suf = ' '.join(suf.split('_'))
    item.name = f'{pref}::{suf}'

