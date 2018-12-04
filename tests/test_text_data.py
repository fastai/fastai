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

def text_files(path, labels):
    os.makedirs(path/'temp', exist_ok=True)
    texts = ["fast ai is a cool project", "hello world"] * 20
    for lbl in labels:
        os.makedirs(path/'temp'/lbl, exist_ok=True)
        for i,t in enumerate(texts):
            with open(path/'temp'/lbl/f'{lbl}_{i}.txt', 'w') as f: f.write(t)

def test_from_folder():
    path = untar_data(URLs.IMDB_SAMPLE)
    text_files(path, ['pos', 'neg'])
    data = (TextList.from_folder(path/'temp')
               .random_split_by_pct(0.1)
               .label_from_folder()
               .databunch())
    assert (len(data.train_ds) + len(data.valid_ds)) == 80
    assert set(data.classes) == {'neg', 'pos'}
    shutil.rmtree(path/'temp')


def special_fastai_test_rule(s): return s.replace("fast ai", "@fastdotai")

def test_from_csv_and_from_df():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos']) #"fast ai is a cool project", "hello world"
    data1 = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=0, text_cols=["text"])
    assert len(data1.classes) == 2

    df = text_df(['neg','pos','neg pos'])
    data2 = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df,
                                  label_cols=0, text_cols=["text"], label_delim=' ',
                                  tokenizer=Tokenizer(pre_rules=[special_fastai_test_rule]))
    assert len(data2.classes) == 2
    x,y = data2.train_ds[0]
    assert len(y.data) == 2
    assert '@fastdotai' in data2.train_ds.vocab.itos,  "custom tokenzier not used by TextClasDataBunch"
    text_csv_file(path/'tmp.csv', ['neg','pos'])
    data3 = TextLMDataBunch.from_csv(path, 'tmp.csv', test='tmp.csv', label_cols=0, text_cols=["text"])
    assert len(data3.classes) == 1
    data4 = TextLMDataBunch.from_csv(path, 'tmp.csv', test='tmp.csv', label_cols=0, text_cols=["text"], max_vocab=5)
    assert 5 <= len(data4.train_ds.vocab.itos) <= 5+8 # +(8 special tokens - UNK/BOS/etc)

    # Test that the tokenizer parameter is used in from_csv
    data4 = TextLMDataBunch.from_csv(path, 'tmp.csv', test='tmp.csv', label_cols=0, text_cols=["text"],
                                     tokenizer=Tokenizer(pre_rules=[special_fastai_test_rule]))
    assert '@fastdotai' in data4.train_ds.vocab.itos, "It seems that our custom tokenzier was not used by TextClasDataBunch"

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

def test_sortish_sampler():
    ds = [1,2,3,4,5,6,7,8,9,10]
    train_sampler = SortishSampler(ds, key=lambda t: ds[t], bs=2)
    assert len(train_sampler) == 10
    ds_srt = [ds[i] for i in train_sampler]
    assert ds_srt[0] == 10

    # test on small datasets
    ds = [1, 10]
    train_sampler = SortishSampler(ds, key=lambda t: ds[t], bs=2)
    assert len(train_sampler) == 2
    ds_srt = [ds[i] for i in train_sampler]
    assert ds_srt[0] == 10

def test_from_ids_works_for_equally_length_sentences():
    ids = [np.array([0])]*10
    lbl = [0]*10
    data = TextClasDataBunch.from_ids('/tmp', vocab=Vocab({0: BOS, 1:PAD}),
                                      train_ids=ids, train_lbls=lbl,
                                      valid_ids=ids, valid_lbls=lbl, classes={0:0})
    text_classifier_learner(data).fit(1)

def test_from_ids_works_for_variable_length_sentences():
    ids = [np.array([0]),np.array([0,1])]*5 # notice diffrent number of elements in arrays
    lbl = [0]*10
    data = TextClasDataBunch.from_ids('/tmp', vocab=Vocab({0: BOS, 1:PAD}),
                                      train_ids=ids, train_lbls=lbl,
                                      valid_ids=ids, valid_lbls=lbl, classes={0:0})
    text_classifier_learner(data).fit(1)
