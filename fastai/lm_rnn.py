import warnings
from .imports import *
from .torch_imports import *
from .rnn_reg import LockedDropout,WeightDrop,EmbeddingDropout
from .model import Stepper


def seq2seq_reg(output, xtra, loss, alpha=0, beta=0):
    hs,dropped_hs = xtra
    if alpha:  # Activation Regularization
        loss = loss + sum(alpha * dropped_hs[-1].pow(2).mean())
    if beta:   # Temporal Activation Regularization (slowness)
        h = hs[-1]
        if len(h)>1: loss = loss + sum(beta * (h[1:] - h[:-1]).pow(2).mean())
    return loss


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)


class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM layers to drive the network, and
        - variational dropouts in the embedding and LSTM layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, bs, ntoken, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                nhid (int): number of hidden activation per LSTM layer
                nlayers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.

            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else nhid, (nhid if l != nlayers - 1 else emb_sz)//self.ndir,
             1, bidirectional=bidir, dropout=dropouth) for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.bs,self.emb_sz,self.nhid,self.nlayers,self.dropoute = bs,emb_sz,nhid,nlayers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (batch_size x sentence length)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()

        emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1: raw_output = drop(raw_output)
            outputs.append(raw_output)

        self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz)//self.ndir
        return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]


class MultiBatchRNN(RNN_Encoder):
    def __init__(self, max_sl, bptt, *args, **kwargs):
        self.max_sl,self.bptt = max_sl,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        raw_outputs, outputs = [],[]
        max_sl = min(self.max_sl,sl)
        for i in range(0, max_sl, self.bptt):
            r, o = super().forward(input[i : min(i+self.bptt, max_sl)])
            #if (i<self.bptt*8) or i>(max_sl-self.bptt*8):
            #if i>(max_sl-self.bptt*16):
            if True:
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)


class LinearRNNOutput(nn.Module):
    initrange=0.1
    def __init__(self, n_out, nhid, dropout):
        super().__init__()
        self.decoder = nn.Linear(nhid, n_out)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        return output, raw_outputs, outputs


class LinearDecoder(LinearRNNOutput):
    """ A custom Linear layer that reads the signals from the output of the RNN_Encoder layer,
    and decodes to a output of size n_tokens.
    """
    def __init__(self, n_out, nhid, dropout, tie_encoder=None):
        super().__init__(n_out, nhid, dropout)
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        output, raw_outputs, outputs = super().forward(input)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x): return self.lin(self.drop(self.bn(x)))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()


def get_language_model(bs, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True):
    """Returns a SequentialRNN model.

    A RNN_Encoder layer is instantiated using the parameters provided.

    This is followed by the creation of a LinearDecoder layer.

    Also by default (i.e. tie_weights = True), the embedding matrix used in the RNN_Encoder
    is used to  instantiate the weights for the LinearDecoder layer.

    The SequentialRNN layer is the native torch's Sequential wrapper that puts the RNN_Encoder and
    LinearDecoder layers sequentially in the model.

    Args:
        bs (int): batch size of input data
        ntoken (int): number of vocabulary (or tokens) in the source dataset
        emb_sz (int): the embedding size to use to encode each token
        nhid (int): number of hidden activation per LSTM layer
        nlayers (int): number of LSTM layers to use in the architecture
        pad_token (int): the int value used for padding text.
        dropouth (float): dropout to apply to the activations going from one LSTM layer to another
        dropouti (float): dropout to apply to the input layer.
        dropoute (float): dropout to apply to the embedding layer.
        wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
        tie_weights (bool): decide if the weights of the embedding matrix in the RNN encoder should be tied to the
            weights of the LinearDecoder layer.
    Returns:
        A SequentialRNN model
    """

    rnn_enc = RNN_Encoder(bs, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(n_tok, emb_sz, dropout, tie_encoder=enc))


def get_rnn_classifer(max_sl, bptt, bs, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5):
    rnn_enc = MultiBatchRNN(max_sl, bptt, bs, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops))

