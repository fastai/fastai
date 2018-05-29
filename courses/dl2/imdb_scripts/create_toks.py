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
        tok += tok_
        labels += labels_
    return tok, labels


def create_toks(dir_path, chunksize=24000, n_lbls=1):
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    df_trn = pd.read_csv(dir_path.joinpath('train.csv'), header=None, chunksize=chunksize)
    df_val = pd.read_csv(dir_path.joinpath('val.csv'), header=None, chunksize=chunksize)

    tmp_path = dir_path.joinpath('tmp')
    tmp_path.mkdir(exist_ok=True)
    tok_trn, trn_labels = get_all(df_trn, n_lbls)
    tok_val, val_labels = get_all(df_val, n_lbls)

    np.save(tmp_path.joinpath('tok_trn.npy'), tok_trn)
    np.save(tmp_path.joinpath('tok_val.npy'), tok_val)
    np.save(tmp_path.joinpath('lbl_trn.npy'), trn_labels)
    np.save(tmp_path.joinpath('lbl_val.npy'), val_labels)

    trn_joined = [' '.join(o) for o in tok_trn]
    open(tmp_path.joinpath('joined.txt'), 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__': fire.Fire(create_toks)
