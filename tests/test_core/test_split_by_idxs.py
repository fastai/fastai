import pytest
from fastai.core import split_by_idxs

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