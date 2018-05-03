from fastai.core import partition
import pytest

def test_partition_functionality():
  szs = [1,2,3,4,5,6]
  a = [1,2,3,4]
  for sz in szs:
    result = partition(a, sz)
    for i, e in enumerate(result):
      assert isinstance(e, list)
      if sz == 1:
        assert len(e) == sz
      elif sz > len(a):
        assert len(e) == len(a)
      elif i == len(result) - 1 and len(a) % sz !=0:
        assert len(e) == len(a) % sz
      else:
        assert len(e) == sz

def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    partition(a, sz)