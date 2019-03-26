import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.tabular import *
from fastai.train import ClassificationInterpretation

pytestmark = pytest.mark.integration
path = untar_data(URLs.ADULT_SAMPLE)

@pytest.fixture(scope="module")
def learn():
    df = pd.read_csv(path/'adult.csv')
    procs = [FillMissing, Categorify, Normalize]
    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    cont_names = ['age', 'fnlwgt', 'education-num']
    test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
    data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_idx(list(range(800,1000)))
            .label_from_df(cols=dep_var)
            .add_test(test)
            .databunch(num_workers=1))
    learn = tabular_learner(data, layers=[200,100], emb_szs={'native-country': 10}, metrics=accuracy)
    learn.fit_one_cycle(2, 1e-2)
    return learn

def test_accuracy(learn):
    this_tests(validate)
    assert learn.validate()[1] > 0.7

def test_same_categories(learn):
    this_tests('na')
    x_train,y_train = learn.data.train_ds[0]
    x_valid,y_valid = learn.data.valid_ds[0]
    x_test,y_test = learn.data.test_ds[0]
    assert x_train.classes.keys() == x_valid.classes.keys()
    assert x_train.classes.keys() == x_test.classes.keys()
    for key in x_train.classes.keys():
        assert np.all(x_train.classes[key] == x_valid.classes[key])
        assert np.all(x_train.classes[key] == x_test.classes[key])

def test_same_fill_nan(learn):
    this_tests('na')
    df = pd.read_csv(path/'adult.csv')
    nan_idx = np.where(df['education-num'].isnull())
    val = None
    for i in nan_idx[0]:
        x,y = (learn.data.train_ds[i] if i < 800 else learn.data.valid_ds[i-800])
        j = x.names.index('education-num') - len(x.cats)
        if val is None: val = x.conts[j]
        else: assert val == x.conts[j]
        if i >= 800:
            x,y = learn.data.test_ds[i-800]
            assert val == x.conts[j]

def test_normalize(learn):
    df = pd.read_csv(path/'adult.csv')
    train_df = df.iloc[0:800].append(df.iloc[1000:])
    c = 'age'
    this_tests('na')
    mean, std = train_df[c].mean(), train_df[c].std()
    for i in np.random.randint(0,799, (20,)):
        x,y = learn.data.train_ds[i]
        assert np.abs(x.conts[0] - (df.loc[i, c] - mean) / (1e-7 + std)) < 1e-6
    for i in np.random.randint(800,1000, (20,)):
        x,y = learn.data.valid_ds[i-800]
        assert np.abs(x.conts[0] - (df.loc[i, c] - mean) / (1e-7 + std)) < 1e-6
    for i in np.random.randint(800,1000, (20,)):
        x,y = learn.data.test_ds[i-800]
        assert np.abs(x.conts[0] - (df.loc[i, c] - mean) / (1e-7 + std)) < 1e-6

def test_empty_cont():
    this_tests('na')
    df = pd.read_csv(path/'adult.csv')
    procs = [FillMissing, Categorify, Normalize]
    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data = (TabularList.from_df(df, path=path, cat_names=cat_names, procs=procs)
            .split_by_idx(list(range(990,1000)))
            .label_from_df(cols=dep_var).databunch(num_workers=1))
    learn = tabular_learner(data, layers=[10], metrics=accuracy)
    learn.fit_one_cycle(1, 1e-1)
    assert learn.validate()[1] > 0.5

def test_confusion_tabular(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    assert isinstance(interp.confusion_matrix(), (np.ndarray))
    assert interp.confusion_matrix().sum() == len(learn.data.valid_ds)
    this_tests(interp.confusion_matrix)
