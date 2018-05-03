from fastai.core import partition
import pytest

def test_partition_functionality():
  sz = 2
  a = [1,2,3,4,5]
  ex = [[1,2],[3,4],[5]]
  result = partition(a, sz)
  assert len(result) == len(ex)
  assert all([a == b for a, b in zip(result, ex)])

  sz = 3
  ex = [[1,2,3],[4,5]]
  result = partition(a, sz)
  assert len(result) == len(ex)
  assert all([a == b for a,b in zip(result, ex)])

  sz = 1
  ex = [[1],[2],[3],[4],[5]]
  result = partition(a, sz)
  assert len(result) == len(ex)
  assert all([a == b for a,b in zip(result, ex)])

  sz = 6
  ex = [[1,2,3,4,5]]
  result = partition(a, sz)
  assert len(result) == len(ex)
  assert all([a == b for a,b in zip(result, ex)])

  sz = 3
  a = []
  result = partition(a, sz)
  assert len(result) == 0

def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    partition(a, sz)
