from fastai.text import *
import html
import fire

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
        texts = texts.apply(fixup).values.astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
        texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


def create_toks(prefix, pr_abbr, chunksize=24000, n_lbls=1):
    PATH = f'data/nlp_clas/{prefix}/'

    df_trn = pd.read_csv(f'{PATH}train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(f'{PATH}test.csv', header=None, chunksize=chunksize)
    print(prefix)

    os.makedirs(f'{PATH}tmp', exist_ok=True)
    tok_trn, trn_labels = get_all(df_trn, n_lbls)
    tok_val, val_labels = get_all(df_val, n_lbls)

    np.save(f'{PATH}tmp/tok_trn.npy', tok_trn)
    np.save(f'{PATH}tmp/tok_val.npy', tok_val)
    np.save(f'{PATH}tmp/lbl_trn.npy', trn_labels)
    np.save(f'{PATH}tmp/lbl_val.npy', val_labels)

    trn_joined = [' '.join(o) for o in tok_trn]
    mdl_fn = f'{PATH}tmp/{pr_abbr}_joined.txt'
    open(mdl_fn, 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__': fire.Fire(create_toks)
