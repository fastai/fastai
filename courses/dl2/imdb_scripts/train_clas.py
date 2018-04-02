import fire
from fastai.text import *
from fastai.lm_rnn import *


def freeze_all_but(learner, n):
    c=learner.get_layer_groups()
    for l in c: set_trainable(l, False)
    set_trainable(c[n], True)


def train_clas(prefix, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
               lr=0.01, dropmult=1.0, pretrain=True, bpe=False, use_clr=True,
               use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
               from_scratch=False, train_file_id=''):
    if clas_id is None: clas_id = lm_id
    print(f'prefix {prefix}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
          f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; pretrain {pretrain}; bpe {bpe}; use_clr {use_clr};'
          f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
          f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id}')
    torch.cuda.set_device(cuda_id)
    PRE = 'bwd_' if backwards else 'fwd_'
    if bpe: PRE = 'bpe_' + PRE
    IDS = 'bpe' if bpe else 'ids'
    if train_file_id != '': train_file_id = f'_{train_file_id}'
    PATH=f'data/nlp_clas/{prefix}/'
    if lm_id != '': lm_id += '_'
    if clas_id != '': clas_id += '_'
    lm_path=f'{PRE}{lm_id}lm_enc'
    assert os.path.exists(os.path.join(PATH, 'models', lm_path + '.h5')),\
        'Error: %s does not exist.' % os.path.join(PATH, 'models', lm_path + '.h5')
    bptt,em_sz,nh,nl = 70,400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_sent = np.load(f'{PATH}tmp/trn_{IDS}{train_file_id}_bwd.npy')
        val_sent = np.load(f'{PATH}tmp/val_{IDS}_bwd.npy')
    else:
        trn_sent = np.load(f'{PATH}tmp/trn_{IDS}{train_file_id}.npy')
        val_sent = np.load(f'{PATH}tmp/val_{IDS}.npy')

    trn_lbls = np.load(f'{PATH}tmp/lbl_trn{train_file_id}.npy')
    val_lbls = np.load(f'{PATH}tmp/lbl_val.npy')
    trn_lbls -= trn_lbls.min()
    val_lbls -= val_lbls.min()
    c=int(trn_lbls.max())+1

    if bpe: vs=30002
    else:
        itos = pickle.load(open(f'{PATH}tmp/itos.pkl', 'rb'))
        vs = len(itos)

    trn_ds = TextDataset(trn_sent, trn_lbls)
    val_ds = TextDataset(val_sent, val_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
    #dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
    #dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
    #dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

    m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25.
    learn.metrics = [accuracy]

    lrm = 2.6
    if use_discriminative:
        lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
    else:
        lrs = lr
    wd = 1e-6
    if not from_scratch:
        learn.load_encoder(lm_path)
    else:
        print('Training classifier from scratch. LM encoder is not loaded.')
        use_regular_schedule = True

    if (startat<1) and pretrain and not last and not chain_thaw and not from_scratch:
        learn.freeze_to(-1)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        learn.freeze_to(-2)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.save(f'{PRE}{clas_id}clas_0')
    elif startat==1:
        learn.load(f'{PRE}{clas_id}clas_0')

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
        print('Using chain-thaw. Unfreezing all layers one at a time...')
        n_layers = len(learn.get_layer_groups())
        print('# of layers:', n_layers)
        # fine-tune last layer
        learn.freeze_to(-1)
        print('Fine-tuning last layer...')
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        n = 0
        # fine-tune all layers up to the second-last one
        while n < n_layers-1:
            print('Fine-tuning layer #%d.' % n)
            freeze_all_but(learn, n)
            learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                      use_clr=None if use_regular_schedule or not use_clr else (8,3))
            n += 1

    if unfreeze:
        learn.unfreeze()
    else:
        learn.freeze_to(-3)

    if last:
        print('Fine-tuning only the last layer...')
        learn.freeze_to(-1)

    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles = cl
        cl = None
    else:
        n_cycles = 1
    learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8,8) if use_clr else None)
    print('Plotting lrs...')
    learn.sched.plot_lr()
    learn.save(f'{PRE}{clas_id}clas_1')

if __name__ == '__main__': fire.Fire(train_clas)

