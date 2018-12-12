import pytest, torch
import numpy as np
from fastai import *
from tempfile import TemporaryDirectory

def test_cpus(): assert num_cpus() >= 1

@pytest.mark.parametrize("p, q, expected", [
    (None, None, []),
    ('hi', None, ['hi']),
    ([1,2],None, [1,2]),
    (5  , 1    , [5]),
    (5  , [1,1], [5, 5]),
    ([5], 1    , [5]),
    ([5], [1,1], [5, 5]),
    ("ab"  , "cd"        , ["ab", "ab"]),
    ("ab"  , ["cd", "ef"], ["ab", "ab"]),
    (["ab"], "cd"        , ["ab", "ab"]),
    (["ab"], ["cd", "ef"], ["ab", "ab"]),
])
def test_listify(p, q, expected):
    assert listify(p, q) == expected

def test_ifnone():
    assert ifnone(None, 5) == 5
    assert ifnone(5, None) == 5
    assert ifnone(1, 5)    == 1
    assert ifnone(0, 5)    == 0

def test_chunks():
    ls = [0,1,2,3]
    assert([a for a in chunks(ls, 2)] == [[0,1],[2,3]])
    assert([a for a in chunks(ls, 4)] == [[0,1,2,3]])
    assert([a for a in chunks(ls, 1)] == [[0],[1],[2],[3]])

def test_uniqueify():
    assert uniqueify([1,1,3,3,5]) == [1,3,5]
    assert uniqueify([1,3,5])     == [1,3,5]
    assert uniqueify([1,1,1,3,5]) == [1,3,5]

def test_listy():
    assert is_listy([1,1,3,3,5])      == True
    assert is_listy((1,1,3,3,5))      == True
    assert is_listy([1,"2",3,3,5])    == True
    assert is_listy((1,"2",3,3,5))    == True
    assert is_listy(1)                == False
    assert is_listy("2")              == False
    assert is_listy({1, 2})           == False
    assert is_listy(set([1,1,3,3,5])) == False

def test_tuple():
    assert is_tuple((1,1,3,3,5)) == True
    assert is_tuple([1])         == False
    assert is_tuple(1)           == False

def test_noop():
    assert noop(1) is 1

def test_to_int():
    assert to_int(("1","1","3","3","5")) == [1,1,3,3,5]
    assert to_int([1,"2",3.3,3,5])       == [1,2,3,3,5]
    assert to_int(1)                     == 1
    assert to_int(1.2)                   == 1
    assert to_int("1")                   == 1

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

    with pytest.raises(Exception): arrays_split([0,5],[1, 2, 3, 4, 5])
    with pytest.raises(Exception): arrays_split([0,3],[1, 2, 3, 4, 5], [1, 2, 3, 4])

def test_random_split():
    valid_pct = 0.4
    a = [len(arr) for arr in random_split(valid_pct, [1,2,3,4,5], ['a', 'b', 'c', 'd', 'e'])]
    b = [2, 2]
    assert a == b

    with pytest.raises(Exception): random_split(1.1, [1,2,3])
    with pytest.raises(Exception): random_split(0.1, [1,2,3], [1,2,3,4])

def test_camel2snake():
    a = camel2snake('someString')
    b = 'some_string'
    assert a == b

    c = camel2snake('some2String')
    d = 'some2_string'
    assert c == d

    e = camel2snake('longStringExmpl')
    f = 'long_string_exmpl'
    assert e == f

def test_even_mults():
    a = even_mults(start=1, stop=8, n=4)
    b = array([1.,2.,4.,8.])
    np.testing.assert_array_equal(a,b)

def test_series2cat():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]})
    cols = 'col1','col2'
    series2cat(df,*cols)
    for col in cols:
        assert (df[col].dtypes == 'category')
    assert (df['col3'].dtypes == 'int64')

def test_download_url():
    for link, ext in [(URLs.MNIST_TINY, 'tgz')]:
        url = f'{link}.{ext}'
        path = URLs.LOCAL_PATH/'data'/'tmp'
        try:
            os.makedirs(path, exist_ok=True)
            filepath = path/url2name(url)
            download_url(url, filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            shutil.rmtree(path)

def test_join_paths():
    assert join_path('f') == Path('f')
    assert join_path('f', Path('dir')) == Path('dir/f')
    assert join_paths(['f1','f2']) == [Path('f1'), Path('f2')]
    assert set(join_paths({'f1','f2'}, Path('dir'))) == {Path('dir/f1'), Path('dir/f2')}

def test_df_names_to_idx():
    df = pd.DataFrame({'col1': [1,2], 'col2': [3,4], 'col3':[5,6]})
    assert df_names_to_idx(['col1','col3'], df) == [0, 2]

def test_one_hot():
    assert all(one_hot([0,-1], 5) == np.array([1,0,0,0,1]))

