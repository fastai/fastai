import pytest
from fastai.core import partition, split_by_idxs

def test_partition_functionality():

  def test_partition(a, sz, ex):
    result = partition(a, sz)
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
  result = partition(a, sz)
  assert len(result) == 0


def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    partition(a, sz)


def test_split_by_idxs_functionality():

  seq = [1,2,3,4,5,6]
  
  def test_split_by_idxs(seq, idxs, ex):
    test_result = []
    for item in split_by_idxs(seq, idxs):
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

  gen = split_by_idxs(seq, idxs)
  with pytest.raises(KeyError):
    next(gen)