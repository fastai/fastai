import pytest, torch
import numpy as np
from fastai import core
from unittest import mock
from unittest.mock import Mock

def test_sum_geom():
    assert core.sum_geom(1, 1, 1) == 1
    assert core.sum_geom(1, 1, 3) == 3
    assert core.sum_geom(3, 10, 4) == 3333
    assert core.sum_geom(0, 2, 3) == 0
    assert core.sum_geom(1, 0, 3) == 1
    assert core.sum_geom(1, 2, 0) == 0

def test_map_none():
  def fn(x): return x
  assert core.map_none(None, fn) == None
  assert core.map_none("not none", fn) == "not none"

def test_delistify():
  assert core.delistify([1]) == 1
  assert core.delistify((1)) == 1
  assert core.delistify("non list") == "non list"
  assert core.delistify(object) == object

  with pytest.raises(IndexError):
    assert core.delistify([])

def test_datafy():
  x = Mock(data={})
  assert core.datafy(x) == {}
  assert core.datafy([x]) == [{}]
  assert core.datafy([x, x]) == [{}, {}]

@mock.patch("fastai.core.torch.cuda.HalfTensor")
def test_T(HalfTensorMock):
  tensor = torch.ones([1, 2])
  assert core.T(tensor) is tensor

  array = np.arange(0, 5)
  assert core.T(array.astype(np.int)).type() == "torch.LongTensor"
  assert core.T(array.astype(np.float)).type() == "torch.FloatTensor"

  core.T(array.astype(np.float), half=True)
  HalfTensorMock.assert_called_once()

  with pytest.raises(NotImplementedError):
    assert core.T(array.astype(np.object))

def test_create_variable_passing_Variable_object():
  v = torch.autograd.Variable(core.T(np.arange(0, 3)))
  assert core.create_variable(v, volatile=True) is v

@mock.patch("fastai.core.Variable")
def test_create_variable(VariableMock):
  v = np.arange(0, 3)

  with mock.patch("fastai.core.IS_TORCH_04", True):
    core.create_variable(v, volatile=True)
    assert VariableMock.call_args[1] == {"requires_grad": False}
  
  with mock.patch("fastai.core.IS_TORCH_04", False):
    core.create_variable(v, volatile=True)
    assert VariableMock.call_args[1] == {"requires_grad": False, "volatile": True}

@mock.patch("fastai.core.create_variable")
def test_V_(create_variable_mock):
  core.V_("foo")

  create_variable_mock.assert_called_with('foo', requires_grad=False, volatile=False)

@mock.patch("fastai.core.map_over")
def test_V(map_over_mock):
  core.V("foo")

  assert map_over_mock.call_args[0][0] == 'foo'
  assert type(map_over_mock.call_args[0][1]) == type(lambda:0)

def test_to_np():
  array = np.arange(0, 3).astype(np.float)
  assert core.to_np(array) is array

  tensor = core.T(array)
  result = core.to_np([tensor, tensor])
  np.testing.assert_equal(result[0], array)
  np.testing.assert_equal(result[1], array)

  variable = core.V(array)
  np.testing.assert_equal(core.to_np(variable), array)

def test_noop():
  assert core.noop() is None

def test_partition_functionality():

  def test_partition(a, sz, ex):
    result = core.partition(a, sz)
    assert len(result) == len(ex)
    assert all([a == b for a, b in zip(result, ex)])

  a = [1,2,3,4,5]
  
  sz = 2
  ex = [[1,2],[3,4],[5]]
  test_partition(a, sz, ex)

  sz = 3
  ex = [[1,2,3],[4,5]]
  test_partition(a, sz, ex)

  sz = 1
  ex = [[1],[2],[3],[4],[5]]
  test_partition(a, sz, ex)

  sz = 6
  ex = [[1,2,3,4,5]]
  test_partition(a, sz, ex)

  sz = 3
  a = []
  result = core.partition(a, sz)
  assert len(result) == 0


def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    core.partition(a, sz)


def test_split_by_idxs_functionality():

  seq = [1,2,3,4,5,6]
  
  def test_split_by_idxs(seq, idxs, ex):
    test_result = []
    for item in core.split_by_idxs(seq, idxs):
      test_result.append(item)
    
    assert len(test_result) == len(ex)
    assert all([a == b for a,b in zip(test_result, ex)])
  
  idxs = [2]
  ex = [[1,2],[3,4,5,6]]

  test_split_by_idxs(seq, idxs, ex)
  
  idxs = [1,2]
  ex = [[1],[2],[3,4,5,6]]
  test_split_by_idxs(seq, idxs, ex)

  idxs = [2,4,5]
  ex = [[1,2],[3,4],[5],[6]]
  test_split_by_idxs(seq, idxs, ex)

  idxs = []
  ex = [[1,2,3,4,5,6]]
  test_split_by_idxs(seq, idxs, ex)


def test_split_by_idxs_error_handling():
  seq = [1,2,3,4]
  idxs = [5]

  gen = core.split_by_idxs(seq, idxs)
  with pytest.raises(KeyError):
    next(gen)