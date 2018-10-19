import pytest
from fastai import *
from fastai.text import *

class DummyTokenizer(BaseTokenizer):
    
    def __init__(self, lang): pass
    def tokenizer(self, texts): return texts.split(' ')
    def add_special_cases(self, toks): pass

#@pytest.mark.slow
@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.IMDB_SAMPLE)
    for file in ['train_tok.npy', 'valid_tok.npy']:
        if os.path.exists(path/'tmp'/file): os.remove(path/'tmp'/file)
    df = pd.read_csv(path/'train.csv', header=None)
    data_lm = TextLMDataBunch.from_df(path, df[:200], df[200:300], tokenizer=Tokenizer(DummyTokenizer))
    data_clas = TextClasDataBunch.from_df(path, df[:200], df[200:300], vocab=data_lm.train_ds.vocab, tokenizer=Tokenizer(DummyTokenizer))
    learn = RNNLearner.classifier(data_clas, emb_sz=100, nl=1, drop_mult=0.1)
    learn.unfreeze()
    learn.fit_one_cycle(1, 1e-2)
    learn = RNNLearner.language_model(data_lm, emb_sz=100, nl=1, drop_mult=0.1)
    learn.fit_one_cycle(2, 1e-2)
    return learn

def test_val_loss(learn):
    assert learn.validate()[1] > 0.25