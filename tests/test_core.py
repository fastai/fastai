from fastai.core import partition
import pytest

def test_partition_functionality():
  # szs = [1,2,3,4,5,6]
  szs = [1,2,3,4,5,6]
  # szs=[2]
  # a = [1,2,3,4,5,6,7,8,9]
  a = []
  # sz = 2
  for sz in szs:
    result = partition(a, sz)
    print('result ---', partition(a, sz), 'SZ ---', sz)
    print('result length ---', len(result))
    for i, e in enumerate(result):
      assert isinstance(e, list)
      if sz == 1:
        assert len(e) == sz
      elif sz > len(a):
        print('oversize element ---', e)
        assert len(e) == len(a)
      elif i == len(result) - 1 and len(a) % sz !=0:
      # elif len(a) % sz !=0:  
        # print(len(e), len(result) % sz)
        print('length of element --', len(e))
        print('modulus ---', len(a) % sz)
        assert len(e) == len(a) % sz
      else:
        print('not at end ---', len(e))
        assert len(e) == sz
  
# def test_partition_on_even():
#   szs = [1,2,3]
#   a = [1,2,3,4]
#   for sz in szs:
#     result = partition(a, sz)
#     print('result ---', partition(a, sz))
#     print('result length ---', len(result))
#     for i,e in enumerate(result):
#       assert isinstance(e, list)


def test_partition_error_handling():
  sz=0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    partition(a, sz)