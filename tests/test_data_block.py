import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.basics import *
from fastai.vision import ImageList


def chk(a,b): assert np.array_equal(a,b)

def test_category():
    this_tests('na')
    c1 = [1,3,2,3,1]
    c2 = list('cabbc')
    df = pd.DataFrame(dict(c1=c1,c2=c2))
    trn_idx = [0,1,3]

    l1 = ItemList.from_df(df, cols=0)
    assert l1[0]==c1[0]
    chk(l1[[1,2]].items, array(c1)[[1,2]])

    sd = l1.split_by_idx([2,4])
    chk(sd.train.items, array(c1)[[0,1,3]])

    ll = sd.label_from_df(1)
    x,y = ll.train.x,ll.train.y
    c2i = {v:k for k,v in enumerate(ll.train.classes)}

    chk(x.items, array(c1)[trn_idx])
    chk(ll.train.classes, ll.valid.classes)
    assert set(ll.train.classes)==set(c2)
    chk(list(map(str, y)), array(c2)[trn_idx])
    chk([o.obj for o in y], array(c2)[trn_idx])
    exp = [c2i[o] for o in array(c2)[trn_idx]]
    chk (list(map(int, y)), exp)
    chk([o.data for o in y], exp)

    c = list('abcd')
    ll = sd.label_from_df(1, classes=c)

    x,y = ll.train.x,ll.train.y
    c2i = {v:k for k,v in enumerate(c)}

    chk(x.items, array(c1)[trn_idx])
    chk(ll.train.classes, ll.valid.classes)
    assert set(ll.train.classes)==set(c)
    chk(list(map(str, y)), array(c2)[trn_idx])
    chk([o.obj for o in y], array(c2)[trn_idx])
    exp = [c2i[o] for o in array(c2)[trn_idx]]
    chk (list(map(int, y)), exp)
    chk([o.data for o in y], exp)

def test_multi_category():
    this_tests('na')
    c1 = [1,3,2,3,1]
    c2     = ['c a', 'a b', 'b c', '', 'a']
    c2_exp = ['c;a', 'a;b', 'b;c', '', 'a']
    c2_obj = [['c', 'a'], ['a', 'b'], ['b', 'c'], [], ['a']]
    df = pd.DataFrame(dict(c1=c1,c2=c2))
    trn_idx = [0,1,3]

    l1 = ItemList.from_df(df, cols=0)
    sd = l1.split_by_idx([2,4])

    ll = sd.label_from_df(1, label_delim=' ')
    x,y = ll.train.x,ll.train.y
    c2i = {v:k for k,v in enumerate(ll.train.classes)}

    chk(x.items, array(c1)[trn_idx])
    chk(ll.train.classes, ll.valid.classes)
    assert set(ll.train.classes)==set(list('abc'))
    chk(list(map(str, y)), array(c2_exp)[trn_idx])
    chk([o.obj for o in y], array(c2_obj)[trn_idx])
    exp = [[c2i[p] for p in o] for o in array(c2_obj)[trn_idx]]
    chk([o.raw for o in y], exp)
    t = c2_obj[1]
    exp = [0.,0.,0.]
    exp[c2i[t[0]]] = 1.
    exp[c2i[t[1]]] = 1.
    chk(y[1].data, exp)

def test_category_processor_existing_class():
    c1 = [1,3,2,3,1]
    c2 = list('cabbc')
    df = pd.DataFrame(dict(c1=c1,c2=c2))

    l1 = ItemList.from_df(df, cols=0)
    sd = l1.split_by_idx([2, 4])
    ll = sd.label_from_df(1)
    ll.y.processor[0].process_one('a')
    this_tests(ll.y.processor[0].process_one)

def test_category_processor_non_existing_class():
    c1 = [1,3,2,3,1]
    c2 = list('cabbc')
    df = pd.DataFrame(dict(c1=c1,c2=c2))

    l1 = ItemList.from_df(df, cols=0)
    sd = l1.split_by_idx([2, 4])
    ll = sd.label_from_df(1)
    this_tests(ll.y.processor[0].process_one)
    assert ll.y.processor[0].process_one('d') is None
    assert ll.y.processor[0].warns == ['d']

def test_splitdata_datasets():
    c1,ratio,n = list('abc'),0.2,10

    this_tests(ItemList.split_by_rand_pct)
    sd = ItemList(range(n)).split_by_rand_pct(ratio).label_const(0)
    assert len(sd.train)==(1-ratio)*n, 'Training set is right size'
    assert len(sd.valid)==ratio*n, 'Validation set is right size'
    assert set(list(sd.train.items)+list(sd.valid.items))==set(range(n)), 'All items covered'

def test_filter_by_rand():
    c1,p,n,seed = list('abc'),0.2,100,759
    
    this_tests(ItemList.filter_by_rand)
    sd1 = ItemList(range(n)).filter_by_rand(p,seed)
    sd2 = ItemList(range(n)).filter_by_rand(p,seed)
    assert len(sd1) == len(sd2), 'Identically seeded random data sets are of different sizes'
    assert (sd1.items == sd2.items).all(), 'Identically seeded random data sets contain different values'

def test_split_subsets():
    this_tests(ItemList.split_subsets)
    sd = ItemList(range(10)).split_subsets(train_size=.1, valid_size=.2).label_const(0)
    assert len(sd.train)==1
    assert len(sd.valid)==2

    with pytest.raises(AssertionError):
        ItemList(range(10)).split_subsets(train_size=.6, valid_size=.6).label_const(0)
    with pytest.raises(AssertionError):
        ItemList(range(10)).split_subsets(train_size=0.0, valid_size=0.5).label_const(0)
    with pytest.raises(AssertionError):
        ItemList(range(10)).split_subsets(train_size=0.5, valid_size=0.0).label_const(0)

def test_regression():
    this_tests('na')
    df = pd.DataFrame({'x':range(100), 'y':np.random.rand(100)})
    data = ItemList.from_df(df, path='.', cols=0).split_by_rand_pct().label_from_df(cols=1).databunch()
    assert data.c==1
    assert isinstance(data.valid_ds, LabelList)

def test_wrong_order():
    this_tests('na')
    path = untar_data(URLs.MNIST_TINY)
    with pytest.raises(Exception, match="Your data isn't split*"):
        ImageList.from_folder(path).label_from_folder().split_by_folder()

class CustomDataset(Dataset):
    def __init__(self, data_list): self.data = copy(data_list)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def test_custom_dataset():
    this_tests(DataBunch)
    tr_dataset = CustomDataset([1, 2, 3])
    val_dataset = CustomDataset([4, 5, 6])
    data = DataBunch.create(tr_dataset, val_dataset)

    # test property fallback
    assert data.loss_func == F.nll_loss

def test_filter_by_folder():
    this_tests(ItemList.filter_by_folder)
    items = ["parent/in", "parent/out", "parent/unspecified_means_out"]

    res = ItemList.filter_by_folder(
        ItemList(items=[Path(p) for p in items], path="parent"),
        include=["in", "and_in"], exclude=["out", "also_out"])
    assert res.items == [Path("parent/in")]

    res = ItemList.filter_by_folder(
        ItemList(items=items),
        include=["in", "and_in"], exclude=["out", "also_out"])
    assert res.items == ["parent/in"]
