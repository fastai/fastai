import pytest
from fastai import *
from fastai.text import *

pytestmark = pytest.mark.integration

def read_file(fname):
    texts = []
    with open(fname, 'r') as f:
        texts = f.readlines()
    labels = [0] * len(texts)
    df = pd.DataFrame({'labels':labels, 'texts':texts}, columns = ['labels', 'texts'])
    return df

def prep_human_numbers():
    path = untar_data(URLs.HUMAN_NUMBERS)
    df_trn = read_file(path/'train.txt')
    df_val = read_file(path/'valid.txt')
    return path, df_trn, df_val

@pytest.fixture(scope="module")
def learn():
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.)
    learn.fit_one_cycle(4, 5e-3)
    return learn

def test_val_loss(learn):
    assert learn.validate()[1] > 0.2

def text_df(n_labels):
    data = []
    texts = ["fast ai is a cool project", "hello world"]
    for ind, text in enumerate(texts):
        sample = {}
        for label in range(n_labels): sample[label] = ind%2
        sample["text"] = text
        data.append(sample)
    df = pd.DataFrame(data)
    return df

def test_classifier():
    for n_labels in [1, 8]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
        os.makedirs(path)
        try:
            df = text_df(n_labels=n_labels)
            data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=list(range(n_labels)), text_cols=["text"])
            classifier = text_classifier_learner(data)
            assert last_layer(classifier.model).out_features == n_labels if n_labels > 1 else n_labels+1
        finally:
            shutil.rmtree(path)

# XXX: may be move into its own test module?
import gc
# everything created by this function should be freed at its exit
def clean_destroy_block():
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.)
    learn.lr_find()

@pytest.mark.skip(reason="memory leak to be fixed")
def test_mem_leak():
    gc.collect()
    garbage_before = len(gc.garbage)  # should be 0 already, or something leaked earlier
    assert garbage_before == 0
    clean_destroy_block()
    gc_collected = gc.collect() # should be 0 too - !0 means we have circular references
    assert gc_collected == 0
    garbage_after = len(gc.garbage)  # again, should be 0, or == garbage_before
    assert garbage_after == 0
