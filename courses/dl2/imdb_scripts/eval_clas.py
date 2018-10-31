import fire
from fastai.text import *
from fastai.lm_rnn import *
from sklearn.metrics import confusion_matrix

def eval_clas(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'lbl_val.npy').flatten()
    val_lbls = val_lbls.flatten()
    val_lbls -= val_lbls.min()
    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(itos)

    m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                layers=[em_sz*3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load_encoder(lm_file)
    learn.load(final_clas_file)
    predictions = np.argmax(learn.predict(), axis=1)
    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls_sampled, predictions))

if __name__ == '__main__': fire.Fire(eval_clas)

