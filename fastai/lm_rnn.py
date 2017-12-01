import warnings
from .imports import *
from .torch_imports import *
from .rnn_reg import embedded_dropout,LockedDropout,WeightDrop
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

    """Instantiates a simple encoder network based on three main principles:

        1. project the input (tokens) on an embedding matrix
        2. use a LSTM (a kind of RNN) to drive the network.
        3. Usage of dropouts in embedding and internal LSTM layers.

    """
    initrange=0.1

    def __init__(self, bs, ntoken, emb_sz, nhid, nlayers, pad_token,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5):

    """ Default constructor for the RNN_Encoder class

        Args:
              bs (int): batch size of input data
              ntoken (int): number of vocabulary (or tokens) in the source dataset
              emb_sz (int): the embedding size to use to encode the input data
              nhid (int): number of hidden activation per layer
              nlayers (int): number of layers
              pad_token (int): the int value used for padding text.
              dropouth (float): dropout to apply to the hidden LSTM layers. Also used to create a custom LockedDropout
                    module which is used on the output.
              dropouti (float): dropout to apply to the input layer. Also used by creating a custom LockedDropout module
              dropoute (float): dropout to apply to the embedding layer.
              wdrop (float): dropout used to instantiate a WeightDrop module.

          Note:
              A module as referenced in this doc is synonymous to a layer created by extending pytorch's torch.nn.Module
              class.
        """
        super().__init__()
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.rnns = [torch.nn.LSTM(emb_sz if l == 0 else nhid, nhid if l != nlayers - 1 else emb_sz, 1, dropout=dropouth)
                     for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.bs,self.emb_sz,self.nhid,self.nlayers,self.dropoute = bs,emb_sz,nhid,nlayers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouth = LockedDropout(dropouth)

    def forward(self, input):
        """ invoked during the forward propagation of the RNN_Encoder module.

        Args:
            input (Tensor): input of shape (batch_size x sentence length)

        Returns:
            raw_outputs (list (Tensor)), outputs (list(Tensor)): The first is the list of tensor evaluated from the pass
            through the LSTM (or any RNN networks). The second seems to be mostly equivalent to the former, except for
            the value got the last layer. This value seems to be what is got from raw_outputs, but with dropouth applied.
        """
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1: raw_output = self.dropouth(raw_output)
            outputs.append(raw_output)

        self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        return Variable(self.weights.new(1, self.bs, self.nhid if l != self.nlayers - 1 else self.emb_sz).zero_(),
                        volatile=not self.training)

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
        if bs==self.bs:
            for l in self.hidden:
                for h in l: h.data.zero_()
        else:
            self.bs=bs
            self.reset()
        raw_outputs, outputs = [],[]
        for i in range(0, min(self.max_sl,sl), self.bptt):
            r, o = super().forward(input[i : min(i+self.bptt, sl)])
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
    def __init__(self, n_out, nhid, dropout, tie_encoder=None):
        super().__init__(n_out, nhid, dropout)
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        output, raw_outputs, outputs = super().forward(input)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class PoolingLinearClassifier(LinearRNNOutput):
    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        output, raw_outputs, outputs = super().forward(input)
        bs,_ = output[-1].size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        pooled = torch.cat([output[-1], mxpool, avgpool], 1)
        result = self.decoder(pooled)
        return result, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()


def get_language_model(bs, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True):
    """Returns a SequentialRNN model.

    At first, a RNN_Encoder layer is instantiated using the parameters provided. This layer
    consists of a series of LSTM networks that uses an embedding layer to encode the input tokens,
    as well as multiple dropout layers for regularization.

    This is followed by the creation of a LinearDecoder layer. This is a simple Linear layer that
    reads the signals from the output of the RNN_Encoder layer, and decodes to a output of size n_tokens.

    Also under default condition (i.e. tie_weights = True), the embedding matrix used in the RNN_Encoder
    is used to  instantiate the weights for the LinearDecoder layer.

    The SequentialRNN layer is the native torch's Sequential wrapper that puts the RNN_Encoder and
    LinearDecoder layers sequentially in the model.

    Args:
        bs (int): batch size of input data
        ntoken (int): number of vocabulary (or tokens) in the source dataset
        emb_sz (int): the embedding size to use to encode the input data
        nhid (int): number of hidden activation per layer
        nlayers (int): number of layers
        pad_token (int): the int value used for padding text.
        dropouth (float): dropout to apply to the hidden LSTM layers. Also used to create a custom LockedDropout
            module which is used on the output.
        dropouti (float): dropout to apply to the input layer. Also used by creating a custom LockedDropout module
        dropoute (float): dropout to apply to the embedding layer.
        wdrop (float): dropout used to instantiate a WeightDrop module.
        tie_weights (bool): decide if the weights of the embedding matrix in the RNN encoder should be tied to the
            weights of the LinearDecoder layer.

    Returns:
        A SequentialRNN model

    """
    rnn_enc = RNN_Encoder(bs, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(n_tok, emb_sz, dropout, tie_encoder=enc))


def get_rnn_classifer(max_sl, bptt, bs, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token,
                      dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5):
    rnn_enc = MultiBatchRNN(max_sl, bptt, bs, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop)
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(n_class, 3*emb_sz, dropout))

