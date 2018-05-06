import pytest
from fastai.core import partition

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