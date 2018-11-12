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
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.1)
    learn.fit_one_cycle(4, 5e-3)
    return learn

def manual_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_val_loss(learn):
    assert learn.validate()[1] > 0.5

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="QRNN requires cupy that depends on cuda")
def test_qrnn_works_with_no_split():
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.1, qrnn=True)
    learn = LanguageLearner(data, learn.model, bptt=70) #  remove the split_fn
    learn.fit_one_cycle(4, 5e-3)
    assert learn.validate()[1] > 0.5

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="QRNN requires cupy that depends on cuda")
def test_qrnn_works_if_split_fn_provided():
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, emb_sz=100, nl=1, drop_mult=0.1, qrnn=True) # it sets: split_func=lm_split
    learn.fit_one_cycle(4, 5e-3)
    assert learn.validate()[1] > 0.5


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
