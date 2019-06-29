import pytest
from fastai.tabular import *
from fastai.gen_doc.doctest import this_tests

def test_from_df():
    path = Path('data/adult_sample/')
    datafile = path/'adult.csv'
    assert datafile.exists(), f'We assume test data is in {datafile}'
    df = pd.read_csv(datafile,nrows=5)

    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    procs = [FillMissing, Categorify, Normalize]

    tablist = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names,procs=procs)

    this_tests(tablist.from_df)
    assert tablist.cat_names == cat_names
    assert tablist.cont_names == cont_names
    assert tablist.procs == procs
    assert (tablist.items == df.index).all()
    assert tablist.path == path

    # Check correct initialization and `get`; 3rd record without 'NaN'
    assert (tablist[3] == df.iloc[3]).all()
