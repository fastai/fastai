from fastai.gen_doc.doctest import this_tests
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import pytest

def _run_batch_size_test(bs):
    dataset = TensorDataset(torch.randn(9, 128), torch.randint(0,3,(9,)))
    dataloader = DataLoader(dataset, batch_size=bs)
    simple_model = nn.Sequential(nn.BatchNorm1d(128), nn.Linear(128,4))
    for (x,y) in iter(dataloader):
        z = simple_model(x)

# This test will fail as the last batch will have a size of 1
@pytest.mark.skip
def test_batch_size_4():
    this_tests('na')
    _run_batch_size_test(4)

# This test succeeds
def test_batch_size_3():
    this_tests('na')
    _run_batch_size_test(3)
