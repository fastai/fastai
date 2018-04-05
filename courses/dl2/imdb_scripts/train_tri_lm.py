import fire
from fastai.text import *
from fastai.lm_rnn import *


class EarlyStopping(Callback):
    def __init__(self, learner, save_path, enc_path=None, patience=5):
        super().__init__()
        self.learner=learner
        self.save_path=save_path
        self.enc_path=enc_path
        self.patience=patience
    def on_train_begin(self):
        self.best_val_loss=100
        self.num_epochs_no_improvement=0
    def on_epoch_end(self, metrics):
        val_loss = metrics[0]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            self.learner.save(self.save_path)
            if self.enc_path is not None:
                self.learner.save_encoder(self.enc_path)
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f'Stopping - no improvement after {self.patience+1} epochs')
            return True
    def on_train_end(self):
        print(f'Loading best model from {self.save_path}')
        self.learner.load(self.save_path)


def train_lm(prefix, cuda_id=0, cl=1, pretrain='wikitext-103-nopl', lm_id='', bs=64,
             dropmult=1.0, backwards=False, lr=0.4e-3, preload=True, bpe=False, startat=0,
             use_clr=True, use_regular_schedule=False, use_discriminative=True, notrain=False, joined=False,
             train_file_id='', early_stopping=False, figshare=False):
    print(f'prefix {prefix}; cuda_id {cuda_id}; cl {cl}; bs {bs}; backwards {backwards} '
          f'dropmult {dropmult}; lr {lr}; preload {preload}; bpe {bpe}; startat {startat} '
          f'pretrain {pretrain}; use_clr {use_clr}; notrain {notrain}; joined {joined} '
          f'early stopping {early_stopping}, figshare {figshare}')

    assert not (figshare and joined), 'Use either figshare or joined.'
    torch.cuda.set_device(cuda_id)
    PRE  = 'bwd_' if backwards else 'fwd_'
    if bpe: PRE = 'bpe_' + PRE
    IDS = 'bpe' if bpe else 'ids'
    if train_file_id != '': train_file_id = f'_{train_file_id}'

    def get_joined_id(): return 'lm_' if joined else ''
    joined_id = 'fig_' if figshare else get_joined_id()
    PATH=f'data/nlp_clas/{prefix}/'
    PRETRAIN_PATH=f'data/nlp_clas/{pretrain}'
    assert os.path.exists(PRETRAIN_PATH), 'Error: %s does not exist.' % PRETRAIN_PATH
    PRE_LM_PATH=f'{PRETRAIN_PATH}/models/{PRE}lm_3.h5'
    assert os.path.exists(PRE_LM_PATH), 'Error: %s does not exist.' % PRE_LM_PATH
    if lm_id != '': lm_id += '_'
    lm_path=f'{PRE}{lm_id}lm'
    enc_path=f'{PRE}{lm_id}lm_enc'
    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_lm_path = f'{PATH}tmp/trn_{joined_id}{IDS}{train_file_id}_bwd.npy'
        val_lm_path = f'{PATH}tmp/val_{joined_id}{IDS}_bwd.npy'
    else:
        trn_lm_path = f'{PATH}tmp/trn_{joined_id}{IDS}{train_file_id}.npy'
        val_lm_path = f'{PATH}tmp/val_{joined_id}{IDS}.npy'

    print(f'Loading {trn_lm_path} and {val_lm_path}')
    trn_lm = np.load(trn_lm_path)
    print('Train data shape before concatentation:', trn_lm.shape)
    if figshare:
        print('Restricting train data to 15M documents...')
        trn_lm = trn_lm[:15000000]

    trn_lm = np.concatenate(trn_lm)
    print('Train data shape after concatentation:', trn_lm.shape)
    val_lm = np.load(val_lm_path)
    val_lm = np.concatenate(val_lm)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(f'{PATH}tmp/itos.pkl', 'rb'))
        vs = len(itos)

    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs, bptt)
    md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*dropmult

    learner = md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    learner.metrics = [accuracy]
    wd=1e-7

    lrs = np.array([lr/6,lr/3,lr,lr/2]) if use_discriminative else lr
    if preload and (startat==0):
        wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
        if bpe: learner.model.load_state_dict(wgts)
        else:
            print(f'Using {pretrain} weights...')
            ew = to_np(wgts['0.encoder.weight'])
            row_m = ew.mean(0)

            itos2 = pickle.load(open(f'{PRETRAIN_PATH}/tmp/itos.pkl', 'rb'))
            stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
            nw = np.zeros((vs, em_sz), dtype=np.float32)
            nb = np.zeros((vs,), dtype=np.float32)
            for i,w in enumerate(itos):
                r = stoi2[w]
                if r>=0: nw[i] = ew[r]
                else:    nw[i] = row_m

            wgts['0.encoder.weight'] = T(nw)
            wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
            wgts['1.decoder.weight'] = T(np.copy(nw))
            learner.model.load_state_dict(wgts)
            #learner.freeze_to(-1)
            #learner.fit(lrs, 1, wds=wd, use_clr=(6,4), cycle_len=1)
    elif preload:
        print('Loading LM that was already fine-tuned on the target data...')
        learner.load(lm_path)

    if not notrain:
        learner.unfreeze()
        if use_regular_schedule:
            print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
            use_clr = None
            n_cycles=cl
            cl=None
        else:
            n_cycles=1
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
            print('Using early stopping...')
        learner.fit(lrs, n_cycles, wds=wd, use_clr=(32,10) if use_clr else None, cycle_len=cl,
                    callbacks=callbacks)
        learner.save(lm_path)
        learner.save_encoder(enc_path)
    else:
        print('No more fine-tuning used. Saving original LM...')
        learner.save(lm_path)
        learner.save_encoder(enc_path)

if __name__ == '__main__': fire.Fire(train_lm)

