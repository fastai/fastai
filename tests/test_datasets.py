import pytest
from fastai.datasets import URLs


@pytest.mark.parametrize("dataset", [
    'adult', 'mnist', 'movie_lens',
    # 'imdb',  # imdb fails unless 'en' spacy language is available
])
def test_get_samples(dataset, tmpdir):
    method = f'get_{dataset}'
    df = getattr(URLs, method)()
    assert df is not None
