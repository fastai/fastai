'Model training for NLP'
from ..torch_core import *
from ..basic_train import *
from ..callbacks import *
from ..basic_data import *
from ..datasets import untar_data
from ..metrics import accuracy
from ..train import GradientClipping
from .models import get_language_model, get_rnn_classifier
from .transform import *

__all__ = ['RNNLearner', 'LanguageLearner', 'RNNLearner', 'convert_weights', 'lm_split',
           'rnn_classifier_split', 'language_model_learner', 'text_classifier_learner', 'default_dropout']

default_dropout = {'language': np.array([0.25, 0.1, 0.2, 0.02, 0.15]),
                   'classifier': np.array([0.4,0.5,0.05,0.3,0.4])}

def convert_weights(wgts:Weights, stoi_wgts:Dict[str,int], itos_new:Collection[str]) -> Weights:
    "Convert the model `wgts` to go with a new vocabulary."
    dec_bias, enc_wgts = wgts['1.decoder.bias'], wgts['0.encoder.weight']
    bias_m, wgts_m = dec_bias.mean(0), enc_wgts.mean(0)
    new_w = enc_wgts.new_zeros((len(itos_new),enc_wgts.size(1))).zero_()
    new_b = dec_bias.new_zeros((len(itos_new),)).zero_()
    for i,w in enumerate(itos_new):
        r = stoi_wgts[w] if w in stoi_wgts else -1
        new_w[i] = enc_wgts[r] if r>=0 else wgts_m
        new_b[i] = dec_bias[r] if r>=0 else bias_m
    wgts['0.encoder.weight'] = new_w
    wgts['0.encoder_dp.emb.weight'] = new_w.clone()
    wgts['1.decoder.weight'] = new_w.clone()
    wgts['1.decoder.bias'] = new_b
    return wgts

def lm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    groups = [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups.append([model[0].encoder, model[0].encoder_dp, model[1]])
    return groups

def rnn_classifier_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    groups = [[model[0].encoder, model[0].encoder_dp]]
    groups += [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups.append([model[1]])
    return groups

class RNNLearner(Learner):
    "Basic class for a Learner in RNN."
    def __init__(self, data:DataBunch, model:nn.Module, bptt:int=70, split_func:OptSplitFunc=None, clip:float=None,
                 adjust:bool=False, alpha:float=2., beta:float=1., metrics=None, **kwargs):
        super().__init__(data, model, **kwargs)
        self.callbacks.append(RNNTrainer(self, bptt, alpha=alpha, beta=beta, adjust=adjust))
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if split_func: self.split(split_func)
        self.metrics = ifnone(metrics, [accuracy])

    def save_encoder(self, name:str):
        "Save the encoder to `name` inside the model directory."
        torch.save(self.model[0].state_dict(), self.path/self.model_dir/f'{name}.pth')

    def load_encoder(self, name:str):
        "Load the encoder `name` from the model directory."
        self.model[0].load_state_dict(torch.load(self.path/self.model_dir/f'{name}.pth'))
        self.freeze()

    def load_pretrained(self, wgts_fname:str, itos_fname:str):
        "Load a pretrained model and adapts it to the data vocabulary."
        old_itos = pickle.load(open(itos_fname, 'rb'))
        old_stoi = {v:k for k,v in enumerate(old_itos)}
        wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
        if 'model' in wgts: wgts = wgts['model']
        wgts = convert_weights(wgts, old_stoi, self.data.train_ds.vocab.itos)
        self.model.load_state_dict(wgts)

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None, pbar:Optional[PBar]=None,
                  ordered:bool=False) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        self.model.reset()
        preds = super().get_preds(ds_type=ds_type, with_loss=with_loss, n_batch=n_batch, pbar=pbar)
        if ordered and hasattr(self.dl(ds_type), 'sampler'):
            sampler = [i for i in self.dl(ds_type).sampler]
            reverse_sampler = np.argsort(sampler)
            preds[0] = preds[0][reverse_sampler,:] if preds[0].dim() > 1 else preds[0][reverse_sampler]
            preds[1] = preds[1][reverse_sampler,:] if preds[1].dim() > 1 else preds[1][reverse_sampler]
        return(preds)

class LanguageLearner(RNNLearner):
    "Subclass of RNNLearner for predictions."

    def predict(self, text:str, n_words:int=1, no_unk:bool=True, temperature:float=1., min_p:float=None):
        "Return the `n_words` that come after `text`."
        ds = self.data.single_dl.dataset
        self.model.reset()
        for _ in progress_bar(range(n_words), leave=False):
            xb, yb = self.data.one_item(text)
            xb = xb.view(-1,1)
            res = self.pred_batch(batch=(xb,yb))[-1]
            if no_unk: res[self.data.vocab.stoi[UNK]] = 0.
            if min_p is not None: res[res < min_p] = 0.
            if temperature != 1.: res.div_(temperature)
            idx = torch.multinomial(res, 1).item()
            text += f' {self.data.vocab.itos[idx]}'
        return text

    def show_results(self, ds_type=DatasetType.Valid, rows:int=5, max_len:int=20):
        from IPython.display import display, HTML
        "Show `rows` result of predictions on `ds_type` dataset."
        ds = self.dl(ds_type).dataset
        self.callbacks.append(RecordOnCPU())
        preds = self.pred_batch(ds_type)
        x,y = self.callbacks[-1].input,self.callbacks[-1].target
        self.callbacks = self.callbacks[:-1]
        y = y.view(*x.size())
        z = preds.view(*x.size(),-1).argmax(dim=2)
        xs = [ds.x.reconstruct(grab_idx(x, i, self.data._batch_first)) for i in range(rows)]
        ys = [ds.x.reconstruct(grab_idx(y, i, self.data._batch_first)) for i in range(rows)]
        zs = [ds.x.reconstruct(grab_idx(z, i, self.data._batch_first)) for i in range(rows)]

        items = [['text', 'target', 'pred']]
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            txt_x = ' '.join(x.text.split(' ')[:max_len])
            txt_y = ' '.join(y.text.split(' ')[max_len:2*max_len])
            txt_z = ' '.join(z.text.split(' ')[max_len:2*max_len])
            items.append([str(txt_x), str(txt_y), str(txt_z)])
        display(HTML(text2html_table(items, ([34,33,33]))))

def language_model_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
                  drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False, pretrained_model=None,
                  pretrained_fnames:OptStrTuple=None, **kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data`."
    dps = default_dropout['language'] * drop_mult
    vocab_size = len(data.vocab.itos)
    model = get_language_model(vocab_size, emb_sz, nh, nl, pad_token, input_p=dps[0], output_p=dps[1],
                weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn)
    learn = LanguageLearner(data, model, bptt, split_func=lm_split, **kwargs)
    if pretrained_model is not None:
        model_path = untar_data(pretrained_model, data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn

def text_classifier_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
               drop_mult:float=1., qrnn:bool=False,max_len:int=70*20, lin_ftrs:Collection[int]=None,
               ps:Collection[float]=None, **kwargs) -> 'TextClassifierLearner':
    "Create a RNN classifier from `data`."
    dps = default_dropout['classifier'] * drop_mult
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1]
    ds = data.train_ds
    vocab_size, n_class = len(data.vocab.itos), data.c
    layers = [emb_sz*3] + lin_ftrs + [n_class]
    ps = [dps[4]] + ps
    model = get_rnn_classifier(bptt, max_len, n_class, vocab_size, emb_sz, nh, nl, pad_token,
                layers, ps, input_p=dps[0], weight_p=dps[1], embed_p=dps[2], hidden_p=dps[3], qrnn=qrnn)
    learn = RNNLearner(data, model, bptt, split_func=rnn_classifier_split, **kwargs)
    return learn
