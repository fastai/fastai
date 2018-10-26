"""
Train a classifier on top of a language model trained with `pretrain_lm.py`.
Optionally fine-tune LM before.
"""
import numpy as np
import pickle

from fastai.text import TextLMDataBunch, TextClasDataBunch, RNNLearner
from fastai import fit_one_cycle
from courses.dl2.imdb_scripts.utils import PAD, UNK, read_imdb, PAD_TOKEN_ID

from sacremoses import MosesTokenizer
import fire
from collections import Counter
from pathlib import Path


def new_train_clas(dir_path, lang='en', pretrain_name='wt-103', model_dir='models', qrnn=True,
                   fine_tune=True, clean=True, max_vocab=30000, bs=70, bptt=70):
    dir_path = Path(dir_path)
    model_dir = Path(model_dir)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    assert model_dir.exists(), f'Error: {model_dir} does not exist.'

    if qrnn:
        print('Using QRNNs...')

    if clean:
        # use no preprocessing besides MosesTokenizer
        tmp_dir = dir_path / 'tmp'
        tmp_dir.mkdir(exist_ok=True)
        if not (tmp_dir / 'train_ids.npy').exists():
            trn_path = dir_path / 'train.csv'
            tst_path = dir_path / 'test.csv'
            assert trn_path.exists(), f'Error: {trn_path} does not exist.'
            assert tst_path.exists(), f'Error: {tst_path} does not exist.'
            trn_toks, trn_lbls = read_imdb(trn_path, MosesTokenizer(lang))
            tst_toks, tst_lbls = read_imdb(tst_path, MosesTokenizer(lang))

            # split off validation set if it does not exist
            val_path = dir_path / 'valid.csv'
            if not val_path.exists():
                trn_len = int(len(trn_toks) * 0.9)
                trn_toks, val_toks = trn_toks[:trn_len], trn_toks[trn_len:]
                trn_lbls, val_lbls = trn_lbls[:trn_len], trn_lbls[trn_len:]
            else:
                val_toks, val_lbls = read_imdb(val_path, MosesTokenizer(lang))

            # create the vocabulary
            cnt = Counter(word for example in trn_toks for word in example)
            itos = [o for o, c in cnt.most_common(n=max_vocab)]
            itos.insert(0, PAD)
            itos.insert(0, UNK)
            stoi = {w: i for i, w in enumerate(itos)}
            with open(tmp_dir / 'itos.pkl', 'wb') as f:
                pickle.dump(itos, f)

            trn_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in trn_toks])
            val_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in val_toks])
            tst_ids = np.array([([stoi.get(w, stoi[UNK]) for w in s]) for s in tst_toks])

            print(f'Train size: {len(trn_ids)}. Valid size: {len(val_ids)}. '
                  f'Test size: {len(tst_ids)}.')

            for split, ids, lbl in zip(['train', 'valid', 'test'],
                             [trn_ids, val_ids, tst_ids],
                             [trn_lbls, val_lbls, tst_lbls]):
                np.save(tmp_dir / f'{split}_ids.npy', ids)
                np.save(tmp_dir / f'{split}_lbl.npy', lbl)

        data_lm = TextLMDataBunch.from_id_files(tmp_dir, test='test')
        data_clas = TextClasDataBunch.from_id_files(tmp_dir, test='test')
    else:
        # use fastai peprocessing and tokenization
        data_lm = TextLMDataBunch.from_csv(dir_path)
        data_clas = TextClasDataBunch.from_csv(dir_path, vocab=data_lm.train_ds.vocab)

    if qrnn:
        emb_sz, nh, nl = 400, 1550, 3
    else:
        emb_sz, nh, nl = 400, 1150, 3
    learn = RNNLearner.language_model(
        data_lm, bptt=bptt, emb_sz=emb_sz, nh=nh, nl=nl, qrnn=qrnn,
        pad_token=PAD_TOKEN_ID,
        pretrained_fnames=[f'lstm_{pretrain_name}', f'itos_{pretrain_name}'],
        path=model_dir.parent, model_dir=model_dir.name)

    if fine_tune:
        print('Fine-tuning the language model...')
        learn.unfreeze()
        learn.fit(2, slice(1e-4, 1e-2))

        # save encoder
        learn.save_encoder('enc')

    learn = RNNLearner.classifier(data_clas, bptt=bptt, pad_token=PAD_TOKEN_ID,
                                  path=model_dir.parent, model_dir=model_dir.name,
                                  qrnn=qrnn, emb_sz=emb_sz, nh=nh, nl=nl)

    learn.load_encoder('enc')
    fit_one_cycle(learn, 1, 5e-3, (0.8, 0.7), wd=1e-7)

    learn.freeze_to(-2)
    fit_one_cycle(learn, 1, 5e-3, (0.8, 0.7), wd=1e-7)

    learn.unfreeze()
    fit_one_cycle(learn, 10, 5e-3, (0.8, 0.7), wd=1e-7)


if __name__ == '__main__': fire.Fire(new_train_clas)
