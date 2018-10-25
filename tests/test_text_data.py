import pytest
from fastai import *
from fastai.text import *

@pytest.fixture(scope="module")
def text_df(multilabel=False):
    df = pd.DataFrame([{0: 0, "text": "fast ai is a cool project"}, {0: 1, 'text': "hello world"}])
    return df

@pytest.fixture(scope="module")
def text_dataset():
    df = text_df()
    text_ds = TextDataset.from_df('/tmp/', df, label_cols=[0], txt_cols=["text"], min_freq=0, tokenizer=Tokenizer(BaseTokenizer))
    return text_ds

def text_csv_file(filepath):
    file = open(filepath, 'w', encoding='utf-8')
    df = text_df()
    df.to_csv(filepath, index=False, header=None)
    file.close()
    return file

def test_should_load_backwards_lm():
    # GIVEN
    text_ds = text_dataset()
    # WHEN
    lml = LanguageModelLoader(text_ds, bs=1, backwards=True)
    # THEN
    batch = lml.get_batch(0, 70)

    as_text = [text_ds.vocab.itos[x] for x in batch[0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])

def test_from_csv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
    os.makedirs(path)
    filename = 'text'
    filepath = os.path.join(path, filename+'.csv')
    try:
        text_csv_file(filepath)
        data = TextClasDataBunch.from_csv(path, train=filename, valid=filename)
        assert len(data.classes) == 2
        assert set(data.classes) == set([True, False])
    finally:
        shutil.rmtree(path)

def test_from_df():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
    os.makedirs(path)
    try:
        df = text_df()
        data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=[0], txt_cols=["text"])
        assert len(data.classes) == 2
        assert set(data.classes) == set([True, False])
    finally:
        shutil.rmtree(path)