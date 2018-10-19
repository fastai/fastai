import pytest, torch
import numpy as np
from fastai import *

def test_cpus(): assert num_cpus() >= 1

@pytest.mark.parametrize("p, q, expected", [
    (5  , 1    , [5]),
    (5  , [1,1], [5, 5]),
    ([5], 1    , [5]),
    ([5], [1,1], [5, 5]),
    ("ab"  , "cd"        , ["a", "b"]),
    ("ab"  , ["cd", "ef"], ["a", "b"]),
    (["ab"], "cd"        , ["ab", "ab"]),
    (["ab"], ["cd", "ef"], ["ab", "ab"]),
])
def test_listify(p, q, expected):
    assert listify(p, q) == expected

def test_ifnone():
    assert ifnone(None, 5) == 5
    assert ifnone(1, 5)    == 1

def test_uniqueify():
    assert uniqueify([1,1,3,3,5]) == [1,3,5]
    assert uniqueify([1,3,5])     == [1,3,5]
    assert uniqueify([1,1,1,3,5]) == [1,3,5]

def test_listy():
    assert is_listy([1,1,3,3,5]) == True
    assert is_listy((1,1,3,3,5)) == True
    assert is_listy(1)           == False

def test_tuple():
    assert is_tuple((1,1,3,3,5)) == True
    assert is_tuple([1])         == False
    assert is_tuple(1)           == False

def test_noop():
    assert noop(1) is 1

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

def test_idx_dict():
    assert idx_dict(np.array([1,2,3]))=={1: 0, 2: 1, 3: 2}
    assert idx_dict([1, 2, 3])=={1: 0, 2: 1, 3: 2}
    assert idx_dict((1, 2, 3))=={1: 0, 2: 1, 3: 2}

def test_find_classes():
    path = Path('./classes_test').resolve()
    os.mkdir(path)
    classes = ['class_0', 'class_1', 'class_2']
    for class_num in classes:
        os.mkdir(path/class_num)
    try:
        assert find_classes(path)==[Path('./classes_test/class_0').resolve(),Path('./classes_test/class_1').resolve(),Path('./classes_test/class_2').resolve()]
    finally:
        shutil.rmtree(path)

def test_arrays_split():
    a = arrays_split([0,3],[1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])
    b = [(array([1, 4]),array(['a', 'd'])), (array([5, 2]),(array(['e','b'])))]
    np.testing.assert_array_equal(a,b)

    c = arrays_split([0,3],[1, 2, 3, 4, 5])
    d = [(array([1, 4]),), (array([5, 2]),)]
    np.testing.assert_array_equal(c,d)
