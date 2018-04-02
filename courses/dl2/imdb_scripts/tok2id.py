from fastai.text import *
import html
import fire

def tok2id(prefix, max_vocab=60000, min_freq=1):
    print(f'prefix {prefix} max_vocab {max_vocab} min_freq {min_freq}')
    PATH=f'data/nlp_clas/{prefix}/'
    trn_tok = np.load(f'{PATH}tmp/tok_trn.npy')
    val_tok = np.load(f'{PATH}tmp/tok_val.npy')

    freq = Counter(p for o in trn_tok for p in o)
    print(freq.most_common(25))
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))

    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

    np.save(f'{PATH}tmp/trn_ids.npy', trn_lm)
    np.save(f'{PATH}tmp/val_ids.npy', val_lm)
    pickle.dump(itos, open(f'{PATH}tmp/itos.pkl', 'wb'))

if __name__ == '__main__': fire.Fire(tok2id)

