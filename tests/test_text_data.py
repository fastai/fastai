import pytest
from fastai import *
from fastai.text import *

def text_df(labels):
    data = []
    texts = ["fast ai is a cool project", "hello world"] * 20
    for ind, text in enumerate(texts):
        sample = {}
        sample["label"] = labels[ind%len(labels)]
        sample["text"] = text
        data.append(sample)
    return pd.DataFrame(data)

def text_csv_file(filepath, labels):
    file = open(filepath, 'w', encoding='utf-8')
    df = text_df(labels)
    df.to_csv(filepath, index=False)
    file.close()
    return file

def test_from_csv_and_from_df():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])
    data1 = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=0, text_cols=["text"])
    assert len(data1.classes) == 2
    df = text_df(['neg','pos','neg pos'])
    data2 = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df,
                                  label_cols=0, text_cols=["text"], label_delim=' ')
    assert len(data2.classes) == 2
    x,y = data2.train_ds[0]
    assert len(y.data) == 2
    text_csv_file(path/'tmp.csv', ['neg','pos'])
    data3 = TextLMDataBunch.from_csv(path, 'tmp.csv', test='tmp.csv', label_cols=0, text_cols=["text"])
    assert len(data3.classes) == 1
    os.remove(path/'tmp.csv')

def test_should_load_backwards_lm():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])
    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"],
                                   bs=1, backwards=True)
    lml = data.train_dl.dl
    lml.data = lml.batchify(np.concatenate([lml.dataset.x.items[i] for i in range(len(lml.dataset))]))
    batch = lml.get_batch(0, 70)
    as_text = [data.train_ds.vocab.itos[x] for x in batch[0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])

def df_test_collate(data):
    x,y = next(iter(data.train_dl))
    assert x.size(0) == 8
    assert x[0,-1] == 1
    
def test_load_and_save_test():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])
    data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=0, text_cols="text")
    data.save()
    data1 = TextClasDataBunch.load(path)
    assert np.all(data.classes == data1.classes)
    assert np.all(data.train_ds.y.items == data1.train_ds.y.items)
    str1 = np.array([str(o) for o in data.train_ds.y])
    str2 = np.array([str(o) for o in data1.train_ds.y])
    assert np.all(str1 == str2)
    shutil.rmtree(path/'tmp')
