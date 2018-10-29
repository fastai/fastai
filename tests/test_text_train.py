import pytest
from fastai import *
from fastai.text import *

pytestmark = pytest.mark.integration

def read_file(fname, sname):
    texts = []
    with open(fname, 'r') as f:
        texts = f.readlines()
    labels = [0] * len(texts)
    df = pd.DataFrame({'labels':labels, 'texts':texts}, columns = ['labels', 'texts'])
    df.to_csv(sname, index=False, header=None)

def prep_human_numbers():
    path = untar_data(URLs.HUMAN_NUMBERS)
    read_file(path/'train.txt', path/'train.csv')
    read_file(path/'valid.txt', path/'valid.csv')
    return path

@pytest.fixture(scope="module")
def learn():
    path = prep_human_numbers()
    data = TextLMDataBunch.from_csv(path, tokenizer=Tokenizer(BaseTokenizer))
    learn = RNNLearner.language_model(data, emb_sz=100, nl=1, drop_mult=0.)
    learn.fit_one_cycle(4, 1e-2)
    return learn

def test_val_loss(learn):
    assert learn.validate()[1] > 0.3

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
            data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=list(range(n_labels)), txt_cols=["text"])
            classifier = RNNLearner.classifier(data)
            assert last_layer(classifier.model).out_features == n_labels if n_labels > 1 else n_labels+1
        finally:
            shutil.rmtree(path)