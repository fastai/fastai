import pytest
from fastai import *
from fastai.collab import *

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def learn():
    ratings = URLs.get_movie_lens()
    ratings.head()
    series2cat(ratings, 'userId','movieId')
    learn = get_collab_learner(ratings, n_factors=50, pct_val=0.2, min_score=0., max_score=5.)
    learn.fit_one_cycle(3, 5e-3, wd=0.1)
    return learn

def test_val_loss(learn):
    assert learn.validate()[0] < 0.75
