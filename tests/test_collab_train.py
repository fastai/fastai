import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.basics import *
from fastai.collab import *

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.ML_SAMPLE)
    ratings = pd.read_csv(path/'ratings.csv')
    series2cat(ratings, 'userId','movieId')
    data = CollabDataBunch.from_df(ratings)
    learn = collab_learner(data, n_factors=50, y_range=(0.,5.))
    learn.fit_one_cycle(3, 5e-3)
    return learn

def test_val_loss(learn):
    this_tests(learn.validate, CollabDataBunch.from_df, collab_learner)
    assert learn.validate()[0] < 0.8
