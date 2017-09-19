import warnings
import torch
import torch.nn as nn
from torch.autograd import Variable
from .rnn_reg import embedded_dropout,LockedDropout,WeightDrop


class Seq2SeqRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=True):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=dropouth) for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights: self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp,self.nhid,self.nlayers = ninp,nhid,nlayers
        self.dropout,self.dropouti,self.dropouth,self.dropoute = dropout,dropouti,dropouth,dropoute

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h: return result, new_hidden, raw_outputs, outputs
        return result, new_hidden

    def one_hidden(self, bsz, l, weight, train):
        return Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_(), volatile=not train)

    def init_hidden(self, bsz, train):
        weight = next(self.parameters()).data
        return [(self.one_hidden(bsz, l, weight, train), self.one_hidden(bsz, l, weight, train)) for l in range(self.nlayers)]

