from fastai.text import *
import fire


def tok2id(dir_path, max_vocab=30000, min_freq=1):
    print(f'dir_path {dir_path} max_vocab {max_vocab} min_freq {min_freq}')
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    tmp_path = p / 'tmp'
    assert tmp_path.exists(), f'Error: {tmp_path} does not exist.'

    trn_tok = np.load(tmp_path / 'tok_trn.npy')
    val_tok = np.load(tmp_path / 'tok_val.npy')

    freq = Counter(p for o in trn_tok for p in o)
    print(freq.most_common(25))
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))

    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

    np.save(tmp_path / 'trn_ids.npy', trn_lm)
    np.save(tmp_path / 'val_ids.npy', val_lm)
    pickle.dump(itos, open(tmp_path / 'itos.pkl', 'wb'))

if __name__ == '__main__': fire.Fire(tok2id)
