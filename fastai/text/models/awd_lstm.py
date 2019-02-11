from ...torch_core import *
from ...layers import *

__all__ = ['EmbeddingDropout', 'LinearDecoder', 'PoolingLinearClassifier', 'AWD_LSTM', 'RNNDropout', 
           'SequentialRNN', 'WeightDropout', 'dropout_mask', 'awd_lstm_lm_split', 'awd_lstm_clas_split',
           'awd_lstm_lm_config', 'awd_lstm_clas_config']

def dropout_mask(x:Tensor, sz:Collection[int], p:float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p:float=0.5):
        super().__init__()
        self.p=p

    def forward(self, x:Tensor)->Tensor:
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class WeightDropout(nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module:nn.Module, weight_p:float, layer_names:Collection[str]=['weight_hh_l0']):
        super().__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args:ArgStar):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()

class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb:nn.Module, embed_p:float):
        super().__init__()
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words:LongTensor, scale:Optional[float]=None)->Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

class AWD_LSTM(nn.Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=1, hidden_p:float=0.2, 
                 input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False):
        super().__init__()
        self.bs,self.qrnn,self.emb_sz,self.n_hid,self.n_layers = 1,qrnn,emb_sz,n_hid,n_layers
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, n_hid if l != n_layers - 1 else emb_sz,
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True,
                                   use_cuda=torch.cuda.is_available()) for l in range(n_layers)]
            for rnn in self.rnns: rnn.linear = WeightDropout(rnn.linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, n_hid if l != n_layers - 1 else emb_sz, 1, batch_first=True) 
                         for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return one_param(self).new(1, self.bs, nh).zero_()
    
    def select_hidden(self, idxs):
        if self.qrnn: self.hidden = [h[:,idxs,:] for h in self.hidden]
        else: self.hidden = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class LinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_out:int, n_hid:int, output_p:float, tie_encoder:nn.Module=None, bias:bool=True):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output)
        return decoded, raw_outputs, outputs

class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.transpose(1,2), (1,)).view(bs,-1)

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[:,-1], mxpool, avgpool], 1)
        x = self.layers(x)
        return x, raw_outputs, outputs

def awd_lstm_lm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    groups = [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    return groups + [[model[0].encoder, model[0].encoder_dp, model[1]]]
    
def awd_lstm_clas_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    groups = [[model[0].module.encoder, model[0].module.encoder_dp]]
    groups += [[rnn, dp] for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    return groups + [[model[1]]]

awd_lstm_lm_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)

awd_lstm_clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, output_p=0.4, 
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)