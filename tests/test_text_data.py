import pytest, tempfile
from fastai.gen_doc.doctest import this_tests
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
    this_tests(TextList.label_from_folder)
    path = untar_data(URLs.IMDB_SAMPLE)
    text_files(path, ['pos', 'neg'])
    data = (TextList.from_folder(path/'temp')
               .split_by_rand_pct(0.1)
               .label_from_folder()
               .databunch())
    assert (len(data.train_ds) + len(data.valid_ds)) == 80
    assert set(data.classes) == {'neg', 'pos'}
    shutil.rmtree(path/'temp')

def test_filter_classes():
    this_tests(TextList.label_from_folder)
    path = untar_data(URLs.IMDB_SAMPLE)
    text_files(path, ['pos', 'neg', 'unsup'])
    with pytest.warns(UserWarning):
        data = (TextList.from_folder(path/'temp')
                 .split_by_rand_pct(0.1)
                 .label_from_folder(classes=['pos', 'neg'])
                 .databunch())
    assert (len(data.train_ds) + len(data.valid_ds)) == 80
    assert set(data.classes) == {'neg', 'pos'}
    shutil.rmtree(path/'temp')

def special_fastai_test_rule(s): return s.replace("fast ai", "@fastdotai")

def test_from_csv_and_from_df():
    this_tests(TextClasDataBunch.from_df, TextClasDataBunch.from_csv)
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos']) #"fast ai is a cool project", "hello world"
    trn_df,val_df,tst_df = df.iloc[:20],df.iloc[20:],df.iloc[:10]
    data1 = TextClasDataBunch.from_df(path, train_df=trn_df, valid_df=val_df, test_df=tst_df, label_cols=0,
                                      text_cols=["text"], no_check=True)
    assert len(data1.classes) == 2
    x,y = next(iter(data1.valid_dl)) # Will fail if the SortSampler keys get messed up between train and valid.
    df = text_df(['neg','pos','neg pos'])
    data2 = TextClasDataBunch.from_df(path, train_df=trn_df, valid_df=val_df,
                                  label_cols=0, text_cols=["text"], label_delim=' ',
                                  tokenizer=Tokenizer(pre_rules=[special_fastai_test_rule]), no_check=True)
    assert len(data2.classes) == 2
    x,y = data2.train_ds[0]
    assert len(y.data) == 2
    assert '@fastdotai' in data2.train_ds.vocab.itos,  "custom tokenzier not used by TextClasDataBunch"
    text_csv_file(path/'tmp.csv', ['neg','pos'])
    data3 = TextLMDataBunch.from_csv(path, 'tmp.csv', test='tmp.csv', label_cols=0, text_cols=["text"], bs=2)
    assert isinstance(data3.train_ds.y[0], EmptyLabel)
    data4 = TextLMDataBunch.from_csv(path, 'tmp.csv', label_cols=0, text_cols=["text"], max_vocab=5, bs=2)
    assert 5 <= len(data4.train_ds.vocab.itos) <= 5+8 # +(8 special tokens - UNK/BOS/etc)
    data4.batch_size = 8

    os.remove(path/'tmp.csv')

def test_should_load_backwards_lm_1():
    "assumes that a backwards batch starts where forward ends. Whether this holds depends on LanguageModelPreLoader"
    path = untar_data(URLs.IMDB_SAMPLE)

    df = text_df(['neg','pos'])
    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"],
                                   bs=2, backwards=False)
    this_tests(data.one_batch)
    batch_forward = data.one_batch(DatasetType.Valid)[0].numpy()

    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"],
                                   bs=2, backwards=True)
    batch_backwards = data.one_batch(DatasetType.Valid)[0].numpy()

    np.testing.assert_array_equal(batch_backwards, np.flip(batch_forward))

def test_should_load_backwards_lm_2():
    "it is fragile to test against specific words. What if 2 batches were split between 'is' an 'a' in df.Text"
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])
    data = TextLMDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=["text"],
                                   bs=2, backwards=True)
    this_tests(data.one_batch)
    batch = data.one_batch(DatasetType.Valid)
    as_text = [data.vocab.itos[x] for x in batch[0][0]]
    np.testing.assert_array_equal(as_text[:2], ["world", "hello"])

def test_backwards_cls_databunch():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg', 'pos'])
    data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=0, text_cols=['text'], bs=4,
                                         backwards=True)
    this_tests(data.one_batch)
    orig_texts = df.text.unique()
    for ds in [DatasetType.Train, DatasetType.Valid]:
        batch = data.one_batch(ds)
        for sample in batch[0]:
            as_text = ' '.join([data.vocab.itos[tok] for tok in sample.flip(0)])
            assert any([orig in as_text for orig in orig_texts])  # batch samples contain BOS and optionally PAD tokens

def df_test_collate(data):
    this_tests('na')
    x,y = next(iter(data.train_dl))
    assert x.size(0) == 8
    assert x[0,-1] == 1

def test_load_and_save_test():
    this_tests(load_data)
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df(['neg','pos'])
    data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, test_df=df, label_cols=0, text_cols="text", bs=10)
    data.save()
    data1 = load_data(path, bs=10)
    assert np.all(data.classes == data1.classes)
    assert np.all(data.train_ds.y.items == data1.train_ds.y.items)
    str1 = np.array([str(o) for o in data.train_ds.y])
    str2 = np.array([str(o) for o in data1.train_ds.y])
    assert np.all(str1 == str2)
    os.remove(path/'data_save.pkl')

def test_sortish_sampler():
    this_tests(SortishSampler)
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
    
def test_sort_sampler():
    this_tests(SortSampler)
    ds = [1,2,3,4,5,6,7,8,9,10]
    train_sampler = SortSampler(ds, key=lambda t: ds[t])
    assert len(train_sampler) == 10
    ds_srt = [ds[i] for i in train_sampler]
    assert ds_srt[0] == 10

    # test on small datasets
    ds = [1, 10]
    train_sampler = SortSampler(ds, key=lambda t: ds[t])
    assert len(train_sampler) == 2
    ds_srt = [ds[i] for i in train_sampler]
    assert ds_srt[0] == 10

def test_from_ids_works_for_equally_length_sentences():
    this_tests(TextClasDataBunch.from_ids)
    ids = [np.array([0])]*10
    lbl = [0]*10
    with tempfile.TemporaryDirectory() as tmp:
        data = TextClasDataBunch.from_ids(tmp, vocab=Vocab({0: BOS, 1:PAD}),
                                          train_ids=ids, train_lbls=lbl,
                                          valid_ids=ids, valid_lbls=lbl, classes={0:0}, bs=8)

def test_from_ids_works_for_variable_length_sentences():
    this_tests(TextClasDataBunch.from_ids)
    ids = [np.array([0]),np.array([0,1])]*5 # notice diffrent number of elements in arrays
    lbl = [0]*10
    with tempfile.TemporaryDirectory() as tmp:
        data = TextClasDataBunch.from_ids(tmp, vocab=Vocab({0: BOS, 1:PAD}),
                                      train_ids=ids, train_lbls=lbl,
                                      valid_ids=ids, valid_lbls=lbl, classes={0:0}, bs=8)

def test_from_ids_exports_classes():
    this_tests(TextClasDataBunch.from_ids)
    ids = [np.array([0])]*10
    lbl = [0]*10
    with tempfile.TemporaryDirectory() as tmp:
        data = TextClasDataBunch.from_ids(tmp, vocab=Vocab({0: BOS, 1:PAD}),
                                      train_ids=ids, train_lbls=lbl,
                                      valid_ids=ids, valid_lbls=lbl,
                                      classes=['a', 'b', 'c'], bs=8)
        data.export(Path(tmp)/'export.pkl')
        empty_data = TextClasDataBunch.load_empty(tmp)
        assert hasattr(empty_data, 'classes')
        assert empty_data.classes == ['a', 'b', 'c'] 

def test_regression():
    this_tests('na')
    path = untar_data(URLs.IMDB_SAMPLE)
    df = text_df([0., 1.])
    data = (TextList.from_df(df, path, cols='text')
             .split_by_rand_pct(0.2)
             .label_from_df(cols='label',label_cls=FloatList)
             .databunch(bs=4))
    assert data.c == 1
    x,y = data.one_batch()
