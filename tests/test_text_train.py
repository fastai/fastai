import pytest
from fastai import *
from fastai.text import *

pytestmark = pytest.mark.integration

def read_file(fname):
    tokens = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            tokens.append(np.array(line[:-1].split(' ')[:-1]))
    return tokens

def prep_human_numbers():
    path = untar_data(URLs.HUMAN_NUMBERS)
    trn_tokens, val_tokens = read_file(path/'train.txt'), read_file(path/'valid.txt')
    np.save(path/'train_tok.npy', trn_tokens)
    np.save(path/'valid_tok.npy', val_tokens)
    return path

@pytest.fixture(scope="module")
def learn():
    path = prep_human_numbers()
    data = TextLMDataBunch.from_tokens(path)
    learn = RNNLearner.language_model(data, emb_sz=100, nl=1, drop_mult=0.)
    learn.fit_one_cycle(4, 1e-2)
    return learn

def test_val_loss(learn):
    assert learn.validate()[1] > 0.3
