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
  # result = partition(a, sz)
  # assert len(result) == len(ex)
  # assert all([a == b for a, b in zip(result, ex)])

  sz = 3
  ex = [[1,2,3],[4,5]]
  test_partition(a, sz, ex)
  # result = partition(a, sz)
  # assert len(result) == len(ex)
  # assert all([a == b for a,b in zip(result, ex)])

  sz = 1
  ex = [[1],[2],[3],[4],[5]]
  # result = partition(a, sz)
  test_partition(a, sz, ex)
  # assert len(result) == len(ex)
  # assert all([a == b for a,b in zip(result, ex)])

  sz = 6
  ex = [[1,2,3,4,5]]
  test_partition(a, sz, ex)
  # result = partition(a, sz)
  # assert len(result) == len(ex)
  # assert all([a == b for a,b in zip(result, ex)])

  sz = 3
  a = []
  result = partition(a, sz)
  assert len(result) == 0

def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    partition(a, sz)

# a generator is a fn that returns an iterable
# to create a generator, define a normal fn with a yield statement instead of a return statement
# while a return statement terminates a fn entirely, a yield statement pauses the fn, saving all its states
# a generator contains one or more yield statements
# when called, it returns an iterator but does not start execution immediately

  # seq = [1,2,3,4,5,6]
  # idx = [2,4]
# --> [1,2], [3,4], [5,6]
# --> split before the second element (3). next, split from the second element up until the 4th (5)

# idx = [2,3]
# --> [1,2], [3], [4,5,6]

# examples with 3 indices:
  # seq = [1,2,3,4,5,6]
  # idx = [2,4,5]
# --> [1,2], [3,4], [5], [6]

# what are the most useful tests for this function?
  # 1) 1 index
  # 2) 2 indices
  # 3) 3 indices
  # 4) no indices
  # 5) all indices

  # for item in enumerate(split_by_idxs(seq, idxs)):
  #   print('item ---', item)
  #   result.append(item)
  # print('result ---', result)
  # assert len(result) == len(ex)
  # assert all([a == b for a,b in zip(result, ex)])

  # assert len(result) == len(ex)
  # assert all([a == b for a,b in zip(result, ex)])


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


















  # one = next(result)
  # print('one ---', one)
  # two = next(result)
  # # test_result.append(one, two)
  # test_result = [one, two]
  # # three = next(result)
  # # four = next(result)
  # # for item, i in enumerate(split_by_idxs(seq, idx)):
  # #   print(' ----', item, 'at index ---', i)
  # print('two ---', two)
  # print('three ---', three)
  # three = next(result)