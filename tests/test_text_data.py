import pytest
from fastai import *
from fastai.text import *

def text_df(n_labels):
    data = []
    texts = ["fast ai is a cool project", "hello world"] * 20
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
    text_ds = (TextDataset.from_df(df, label_cols=[0], txt_cols=["text"])
               .tokenize(tokenizer=Tokenizer(BaseTokenizer))
               .numericalize(min_freq=0))
    # WHEN
    lml = LanguageModelLoader(text_ds, bs=1, backwards=True)
    # THEN
    lml.data = lml.batchify(np.concatenate(lml.dataset.x))
    batch = lml.get_batch(0, 70)
    as_text = [text_ds.vocab.itos[x] for x in batch[0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])

def test_from_csv_and_from_df():
    for func in ['from_csv', 'from_df']:
        for n_labels in [1, 3]:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
            try:
                os.makedirs(path)
                if func is 'from_csv':
                    filename = 'text'
                    text_csv_file(os.path.join(path, filename+'.csv'), n_labels=n_labels)
                    data_bunch = TextDataBunch.from_csv(path, f'{filename}.csv', test=f'{filename}.csv', n_labels=n_labels)
                    clas_data_bunch = TextClasDataBunch.from_csv(path, f'{filename}.csv', test=f'{filename}.csv', n_labels=n_labels)
                else:
                    df = text_df(n_labels=n_labels)
                    data_bunch = TextDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=list(range(n_labels)), txt_cols=["text"])
                    clas_data_bunch = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=list(range(n_labels)), txt_cols=["text"])

                for data in [data_bunch, clas_data_bunch]:
                    assert len(data.classes) == 2 if n_labels==1 else n_labels
                    assert set(data.classes) == {True, False} if n_labels==1 else [1,2,3]
                    if n_labels > 1: assert len(data.y[0]) == n_labels
            finally:
                shutil.rmtree(path)

def test_collate():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv', header=None)
    dft, dfv = df.iloc[:800,:2], df.iloc[800:,:2]
    data = TextClasDataBunch.from_df(path, dft, dfv, bs=20)
    x,y = next(iter(data.train_dl))
    assert x.size(0) == 1519 and x.size(1) == 10
    assert x[0,-1] == 1