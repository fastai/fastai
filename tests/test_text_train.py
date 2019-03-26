import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.text import *

pytestmark = pytest.mark.integration

def read_file(fname):
    texts = []
    with open(fname, 'r') as f: texts = f.readlines()
    labels = [0] * len(texts)
    df = pd.DataFrame({'labels':labels, 'texts':texts}, columns = ['labels', 'texts'])
    return df

def prep_human_numbers():
    path = untar_data(URLs.HUMAN_NUMBERS)
    df_trn = read_file(path/'train.txt')
    df_val = read_file(path/'valid.txt')
    return path, df_trn, df_val

def config(qrnn:bool=False):
    config = awd_lstm_lm_config.copy()
    config['emb_sz'],config['n_hid'],config['n_layers'],config['qrnn'] = 100,100,1,qrnn
    return config

@pytest.fixture(scope="module")
def learn():
    path, df_trn, df_val = prep_human_numbers()
    df = df_trn.append(df_val)
    data = (TextList.from_df(df, path, cols='texts')
                .split_by_idx(list(range(len(df_trn),len(df))))
                .label_for_lm()
                .add_test(df['texts'].iloc[:200].values)
                .databunch())
    learn = language_model_learner(data, AWD_LSTM, pretrained=False, config=config(), drop_mult=0.)
    learn.opt_func = partial(optim.SGD, momentum=0.9)
    learn.fit(3,1)
    return learn

def n_params(learn): return sum([len(pg['params']) for pg in learn.opt.opt.param_groups])

def test_opt_params(learn):
    this_tests('na')
    learn.freeze()
    assert n_params(learn) == 2
    learn.unfreeze()
    assert n_params(learn) == 6

def manual_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_val_loss(learn):
    this_tests(learn.validate)
    assert learn.validate()[1] > 0.3

@pytest.mark.slow
def test_qrnn_works_with_no_split():
    this_tests(language_model_learner)
    gc.collect()
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, AWD_LSTM, pretrained=False, config=config(True), drop_mult=0.)
    learn = LanguageLearner(data, learn.model) #  remove the split_fn
    learn.fit_one_cycle(2, 0.1)
    assert learn.validate()[1] > 0.3

@pytest.mark.slow
def test_qrnn_works_if_split_fn_provided():
    this_tests(language_model_learner)
    gc.collect()
    manual_seed()
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, AWD_LSTM, pretrained=False, config=config(True), drop_mult=0.)
    learn.fit_one_cycle(2, 0.1)
    assert learn.validate()[1] > 0.3

def test_vocabs(learn):
    this_tests('na')
    for ds in [learn.data.valid_ds, learn.data.test_ds]:
        assert len(learn.data.train_ds.vocab.itos) == len(ds.vocab.itos)
        assert np.all(learn.data.train_ds.vocab.itos == ds.vocab.itos)

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
    this_tests(text_classifier_learner)
    for n_labels in [1, 8]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
        os.makedirs(path)
        try:
            expected_classes = n_labels if n_labels > 1 else 2
            df = text_df(n_labels=n_labels)
            data = TextClasDataBunch.from_df(path, train_df=df, valid_df=df, label_cols=list(range(n_labels)), text_cols=["text"], bs=2)
            classifier = text_classifier_learner(data, AWD_LSTM, pretrained=False, bptt=10)
            assert last_layer(classifier.model).out_features == expected_classes
            assert len(data.train_dl) == math.ceil(len(data.train_ds)/data.train_dl.batch_size)
            assert next(iter(data.train_dl))[0].shape == (2, 7)
            assert next(iter(data.valid_dl))[0].shape == (2, 7)
        finally:
            shutil.rmtree(path)

# TODO: may be move into its own test module?
import gc
# everything created by this function should be freed at its exit
def clean_destroy_block():
    path, df_trn, df_val = prep_human_numbers()
    data = TextLMDataBunch.from_df(path, df_trn, df_val, tokenizer=Tokenizer(BaseTokenizer))
    learn = language_model_learner(data, AWD_LSTM, pretrained=False, config=config(), drop_mult=0.)
    learn.lr_find()

@pytest.mark.skip(reason="fix me")
def test_mem_leak():
    this_tests('na')
    gc.collect()
    garbage_before = len(gc.garbage)  # should be 0 already, or something leaked earlier
    assert garbage_before == 0
    clean_destroy_block()

    gc_collected = gc.collect() # should be 0 too - !0 means we have circular references
    assert gc_collected < 102 # scipy has some cyclic references that we want to ignore (this accounts for 100 objects).
    garbage_after = len(gc.garbage)  # again, should be 0, or == garbage_before
    assert garbage_after == 0

def test_order_preds():
    this_tests(text_classifier_learner)
    path, df_trn, df_val = prep_human_numbers()
    df_val.labels = np.random.randint(0,5,(len(df_val),))
    data_clas = (TextList.from_df(df_val, path, cols='texts')
                .split_by_idx(list(range(200)))
                .label_from_df(cols='labels')
                .databunch())
    learn = text_classifier_learner(data_clas, AWD_LSTM, pretrained=False)
    preds = learn.get_preds(ordered=True)
    true_value = np.array([learn.data.train_ds.c2i[o] for o in df_val.iloc[:200,0]])
    np.all(true_value==preds[1].numpy())
