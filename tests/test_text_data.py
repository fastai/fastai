import pytest
from fastai import *
from fastai.text import *

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

def text_csv_file(filepath, n_labels):
    file = open(filepath, 'w', encoding='utf-8')
    df = text_df(n_labels)
    df.to_csv(filepath, index=False, header=None)
    file.close()
    return file

def test_should_load_backwards_lm():
    # GIVEN
    df = text_df(n_labels=1)
    text_ds = TextDataset.from_df('/tmp/', df, label_cols=[0], txt_cols=["text"], min_freq=0, tokenizer=Tokenizer(BaseTokenizer))
    # WHEN
    lml = LanguageModelLoader(text_ds, bs=1, backwards=True)
    # THEN
    batch = lml.get_batch(0, 70)

    as_text = [text_ds.vocab.itos[x] for x in batch[0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])

def test_from_csv():
    for n_labels in [1, 3]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
        os.makedirs(path)
        filename = 'text'
        filepath = os.path.join(path, filename+'.csv')
        try:
            text_csv_file(filepath, n_labels=n_labels)
            data = TextClasDataBunch.from_csv(path, train=filename, valid=filename, n_labels=n_labels)
            assert len(data.classes) == 2
            assert set(data.classes) == set([True, False])
            if n_labels > 1: assert len(data.labels[0]) == n_labels
        finally:
            shutil.rmtree(path)

def test_from_df():
    for n_labels in [1, 3]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
        os.makedirs(path)
        try:
            df = text_df(n_labels=n_labels)
            data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=list(range(n_labels)), txt_cols=["text"])
            assert len(data.classes) == 2
            assert set(data.classes) == set([True, False])
            if n_labels > 1: assert len(data.labels[0]) == n_labels
        finally:
            shutil.rmtree(path)