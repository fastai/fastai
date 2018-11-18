from fastai import *

def chk(a,b): assert np.array_equal(a,b)

def test_category():
    c1 = [1,3,2,3,1]
    c2 = list('cabbc')
    df = pd.DataFrame(dict(c1=c1,c2=c2))
    trn_idx = [0,1,3]

    l1 = ItemList.from_df(df, col=0)
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
    c1 = [1,3,2,3,1]
    c2     = ['c a', 'a b', 'b c', '', 'a']
    c2_exp = ['c;a', 'a;b', 'b;c', '', 'a']
    c2_obj = [['c', 'a'], ['a', 'b'], ['b', 'c'], [], ['a']]
    df = pd.DataFrame(dict(c1=c1,c2=c2))
    trn_idx = [0,1,3]

    l1 = ItemList.from_df(df, col=0)
    sd = l1.split_by_idx([2,4])

    ll = sd.label_from_df(1, sep=' ')
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

