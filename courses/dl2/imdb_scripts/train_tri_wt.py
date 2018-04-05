import fire
from fastai.learner import *
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.text import *
from fastai.lm_rnn import *

from sampled_sm import *


def train_lm(prefix, cuda_id, cl=1, bs=64, backwards=False, lr=3e-4, startat=0, sampled=True, preload=True):
    print(f'prefix {prefix}; cuda_id {cuda_id}; cl {cl}; bs {bs}; backwards {backwards} sampled {sampled} '
          f'lr {lr} startat {startat}')
    torch.cuda.set_device(cuda_id)
    PRE  = 'bwd_' if backwards else 'fwd_'
    PRE2 = PRE
    PRE2 = 'bwd_'
    IDS = 'ids'
    NLPPATH=Path('data/nlp_clas')
    PATH=NLPPATH / prefix
    PATH2=NLPPATH / 'wikitext-103_2'
    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_lm = np.load(PATH / f'tmp/trn_{IDS}_bwd.npy')
        val_lm = np.load(PATH / f'tmp/val_{IDS}_bwd.npy')
    else:
        trn_lm = np.load(PATH / f'tmp/trn_{IDS}.npy')
        val_lm = np.load(PATH / f'tmp/val_{IDS}.npy')
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.concatenate(val_lm)

    itos = pickle.load(open(PATH / 'tmp/itos.pkl', 'rb'))
    vs = len(itos)

    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs//5 if sampled else bs, bptt)
    md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    tprs = get_prs(trn_lm, vs)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.5
    learner,crit = get_learner(drops, 15000, sampled, md, em_sz, nh, nl, opt_fn, tprs)
    wd=1e-7
    learner.metrics = [accuracy]

    if (startat<1) and preload:
        wgts = torch.load(PATH2 / f'models/{PRE2}lm_3.h5', map_location=lambda storage, loc: storage)
        ew = to_np(wgts['0.encoder.weight'])
        row_m = ew.mean(0)

        itos2 = pickle.load(open(PATH2 / 'tmp/itos.pkl', 'rb'))
        stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
        nw = np.zeros((vs, em_sz), dtype=np.float32)
        for i,w in enumerate(itos):
            r = stoi2[w]
            nw[i] = ew[r] if r>=0 else row_m

        wgts['0.encoder.weight'] = T(nw)
        wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
        wgts['1.decoder.weight'] = T(np.copy(nw))
        learner.model.load_state_dict(wgts)
    elif startat==1: learner.load(f'{PRE}lm_4')
    learner.metrics = [accuracy]

    lrs = np.array([lr/6,lr/3,lr,lr])
    #lrs=lr

    learner.unfreeze()
    learner.fit(lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl)
    learner.save(f'{PRE}lm_4')
    learner.save_encoder(f'{PRE}lm_4_enc')

if __name__ == '__main__': fire.Fire(train_lm)

