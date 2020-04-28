import pytest, torch
import numpy as np
from fastai.basics import *
from fastai.gen_doc.doctest import this_tests
from tempfile import TemporaryDirectory
from collections import Counter

def test_cpus():
    this_tests(num_cpus)
    assert num_cpus() >= 1

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
    this_tests(listify)
    assert listify(p, q) == expected

def test_recurse():
    this_tests(recurse)
    def to_plus(x, a=1): return recurse(lambda x,a: x+a, x, a)
    assert to_plus(1) == 2
    assert to_plus([1,2,3]) == [2,3,4]
    assert to_plus([1,2,3], a=3) == [4,5,6]
    assert to_plus({'a': 1, 'b': 2, 'c': 3}) == {'a': 2, 'b': 3, 'c': 4}
    assert to_plus({'a': 1, 'b': 2, 'c': 3}, a=2) == {'a': 3, 'b': 4, 'c': 5}
    assert to_plus({'a': 1, 'b': [1,2,3], 'c': {'d': 4, 'e': 5}}) == {'a': 2, 'b': [2, 3, 4], 'c': {'d': 5, 'e': 6}}

def test_ifnone():
    this_tests(ifnone)
    assert ifnone(None, 5) == 5
    assert ifnone(5, None) == 5
    assert ifnone(1, 5)    == 1
    assert ifnone(0, 5)    == 0

def test_chunks():
    this_tests(chunks)
    ls = [0,1,2,3]
    assert([a for a in chunks(ls, 2)] == [[0,1],[2,3]])
    assert([a for a in chunks(ls, 4)] == [[0,1,2,3]])
    assert([a for a in chunks(ls, 1)] == [[0],[1],[2],[3]])

def test_uniqueify():
    this_tests(uniqueify)
    assert uniqueify([1,1,3,3,5]) == [1,3,5]
    assert uniqueify([1,3,5])     == [1,3,5]
    assert uniqueify([1,1,1,3,5]) == [1,3,5]

def test_listy():
    this_tests(is_listy)
    assert is_listy([1,1,3,3,5])      == True
    assert is_listy((1,1,3,3,5))      == True
    assert is_listy([1,"2",3,3,5])    == True
    assert is_listy((1,"2",3,3,5))    == True
    assert is_listy(1)                == False
    assert is_listy("2")              == False
    assert is_listy({1, 2})           == False
    assert is_listy(set([1,1,3,3,5])) == False

def test_tuple():
    this_tests(is_tuple)
    assert is_tuple((1,1,3,3,5)) == True
    assert is_tuple([1])         == False
    assert is_tuple(1)           == False

def test_dict():
    this_tests(is_dict)
    assert is_dict({1:2,3:4})  == True
    assert is_dict([1,2,3])    == False
    assert is_dict((1,2,3))    == False

def test_noop():
    this_tests(noop)
    assert noop(1) == 1

def test_to_int():
    this_tests(to_int)
    assert to_int(("1","1","3","3","5")) == [1,1,3,3,5]
    assert to_int([1,"2",3.3,3,5])       == [1,2,3,3,5]
    assert to_int(1)                     == 1
    assert to_int(1.2)                   == 1
    assert to_int("1")                   == 1

def test_partition_functionality():
    this_tests(partition)

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
    this_tests(idx_dict)
    assert idx_dict(np.array([1,2,3]))=={1: 0, 2: 1, 3: 2}
    assert idx_dict([1, 2, 3])=={1: 0, 2: 1, 3: 2}
    assert idx_dict((1, 2, 3))=={1: 0, 2: 1, 3: 2}

def test_find_classes():
    this_tests(find_classes)
    path = Path('./classes_test').resolve()
    os.mkdir(path)
    classes = ['class_0', 'class_1', 'class_2']
    for class_num in classes:
        os.mkdir(path/class_num)
    try: assert [o.name for o in find_classes(path)]==classes
    finally: shutil.rmtree(path)

def test_arrays_split():
    this_tests(arrays_split)
    a = arrays_split([0,3],[1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])
    b = [(array([1, 4]),array(['a', 'd'])), (array([5, 2]),(array(['e','b'])))]
    np.testing.assert_array_equal(a,b)

    c = arrays_split([0,3],[1, 2, 3, 4, 5])
    d = [(array([1, 4]),), (array([5, 2]),)]
    np.testing.assert_array_equal(c,d)

    with pytest.raises(Exception): arrays_split([0,5],[1, 2, 3, 4, 5])
    with pytest.raises(Exception): arrays_split([0,3],[1, 2, 3, 4, 5], [1, 2, 3, 4])

def test_random_split():
    this_tests(random_split)
    valid_pct = 0.4
    a = [len(arr) for arr in random_split(valid_pct, [1,2,3,4,5], ['a', 'b', 'c', 'd', 'e'])]
    b = [2, 2]
    assert a == b

    with pytest.raises(Exception): random_split(1.1, [1,2,3])
    with pytest.raises(Exception): random_split(0.1, [1,2,3], [1,2,3,4])

def test_camel2snake():
    this_tests(camel2snake)
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
    this_tests(even_mults)
    a = even_mults(start=1, stop=8, n=4)
    b = array([1.,2.,4.,8.])
    np.testing.assert_array_equal(a,b)

def test_series2cat():
    this_tests(series2cat)
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]})
    cols = 'col1','col2'
    series2cat(df,*cols)
    for col in cols:
        assert (df[col].dtypes == 'category')
    assert (df['col3'].dtypes == 'int64')

def test_download_url():
    this_tests(download_url)
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
    this_tests(join_path)
    assert join_path('f') == Path('f')
    assert join_path('f', Path('dir')) == Path('dir/f')
    assert join_paths(['f1','f2']) == [Path('f1'), Path('f2')]
    assert set(join_paths({'f1','f2'}, Path('dir'))) == {Path('dir/f1'), Path('dir/f2')}

def test_df_names_to_idx():
    this_tests(df_names_to_idx)
    df = pd.DataFrame({'col1': [1,2], 'col2': [3,4], 'col3':[5,6]})
    assert df_names_to_idx(['col1','col3'], df) == [0, 2]

def test_one_hot():
    this_tests(one_hot)
    assert all(one_hot([0,-1], 5) == np.array([1,0,0,0,1]))

def test_subplots_multi_row_cols():
    this_tests(subplots)
    axs = subplots(4, 4, figsize=(10, 10))
    assert len(axs) == 4
    assert (len(axs[0]) == 4)
    assert (len(axs.flatten()) == 16)

def test_subplots_single():
    this_tests(subplots)
    axs = subplots(1,1, figsize=(10, 10))
    assert (len(axs) == 1)
    assert (len(axs[0]) == 1)

def test_is1d():
    this_tests(is1d)
    assert is1d([1, 2, 3, 4])
    assert is1d((1, 2, 3, 4))
    assert not is1d([[1, 2], [3, 4]])
    assert not is1d(np.array(((1,2), (3,4))))

def test_itembase_eq():
    this_tests(ItemBase.__eq__, Category, FloatItem, MultiCategory)
    c1 = Category(0, 'cat')
    c2 = Category(1, 'dog')
    c3 = Category(0, 'cat')
    assert c1 == c1
    assert c1 != c2
    assert c1 == c3

    f1 = FloatItem(0.1)
    f2 = FloatItem(1.2)
    f3 = FloatItem(0.1)
    assert f1 == f1
    assert f1 != f2
    assert f1 == f3

    mc1 = MultiCategory(np.array([1, 0]), ['cat'], [0])
    mc2 = MultiCategory(np.array([1, 1]), ['cat', 'dog'], [0, 1])
    mc3 = MultiCategory(np.array([1, 0]), ['cat'], [0])

    assert mc1 == mc1
    assert mc1 != mc2
    assert mc1 == mc3

    # tensors are used instead of arrays
    mc4 = MultiCategory(torch.Tensor([1, 0]), ['cat'], [0])
    mc5 = MultiCategory(torch.Tensor([1, 1]), ['cat', 'dog'], [0, 1])
    mc6 = MultiCategory(torch.Tensor([1, 0]), ['cat'], [0])

    assert mc4 == mc4
    assert mc4 != mc5
    assert mc4 == mc6

    class TestItemBase(ItemBase):
        def __init__(self, data):
            self.data = data

    # data is a list of objects
    t1 = TestItemBase([torch.Tensor([1, 2]), torch.Tensor([3, 4])])
    t2 = TestItemBase([torch.Tensor([2, 3]), torch.Tensor([3, 4])])
    t3 = TestItemBase([torch.Tensor([1, 2]), torch.Tensor([3, 4])])

    assert t1 == t1
    assert t1 != t2
    assert t1 == t3

    t4 = TestItemBase([1, 2])
    t5 = TestItemBase([1])
    t6 = TestItemBase([1, 2])

    assert t4 == t4
    assert t4 != t5
    assert t4 == t6

    t7 = TestItemBase([[1]])
    t8 = TestItemBase([1])
    t9 = TestItemBase([[1]])

    assert t7 == t7
    assert t7 != t8
    assert t7 == t9

def test_itembase_hash():
    this_tests(ItemBase.__eq__, Category.__hash__, FloatItem.__hash__, MultiCategory.__hash__)

    c1 = Category(0, 'cat')
    c2 = Category(1, 'dog')
    c3 = Category(0, 'cat')
    assert hash(c1) == hash(c3)
    assert hash(c1) != hash(c2)
    assert Counter([c1, c2, c3]) == {c1: 2, c2: 1}

    f1 = FloatItem(0.1)
    f2 = FloatItem(1.2)
    f3 = FloatItem(0.1)
    assert hash(f1) == hash(f3)
    assert hash(f1) != hash(f2)
    assert Counter([f1, f2, f3]) == {f1: 2, f2: 1}

    mc1 = MultiCategory(np.array([1, 0]), ['cat'], [0])
    mc2 = MultiCategory(np.array([1, 1]), ['cat', 'dog'], [0, 1])
    mc3 = MultiCategory(np.array([1, 0]), ['cat'], [0])

    assert hash(mc1) == hash(mc3)
    assert hash(mc1) != hash(mc2)
    assert Counter([mc1, mc2, mc3]) == {mc1: 2, mc2: 1}
