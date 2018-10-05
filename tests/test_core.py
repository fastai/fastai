import pytest, torch
import numpy as np
from fastai import core

def test_cpus():
    assert core.num_cpus() >= 1

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
    assert core.listify(p, q) == expected

def test_ifnone():
    assert core.ifnone(None, 5) == 5
    assert core.ifnone(1, 5)    == 1

def test_uniqueify():
    assert core.uniqueify([1,1,3,3,5]) == [1,3,5]
    assert core.uniqueify([1,3,5])     == [1,3,5]
    assert core.uniqueify([1,1,1,3,5]) == [1,3,5]


def test_listy():
    assert core.is_listy([1,1,3,3,5]) == True
    assert core.is_listy((1,1,3,3,5)) == True
    assert core.is_listy(1)           == False

def test_tuple():
    assert core.is_tuple((1,1,3,3,5)) == True
    assert core.is_tuple([1])         == False
    assert core.is_tuple(1)           == False

def test_noop():
    assert core.noop(1) is 1

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
