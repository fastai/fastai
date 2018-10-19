import pytest
from fastai import *
from fastai.tabular import *

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.ADULT_SAMPLE)
    df = pd.read_csv(path/'adult.csv')
    tfms = [FillMissing, Categorify]
    train_df, valid_df = df[:1024].copy(),df[1024:1260].copy()
    dep_var = '>=50k'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data = TabularDataBunch.from_df(path, train_df, valid_df, dep_var, tfms=tfms, cat_names=cat_names)
    learn = get_tabular_learner(data, layers=[200,100], emb_szs={'native-country': 10}, metrics=accuracy)
    learn.fit_one_cycle(3, 1e-2)
    return learn

def test_accuracy(learn):
    assert learn.validate()[1] > 0.75
