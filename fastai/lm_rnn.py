import warnings
import torch
import torch.nn as nn
from torch.autograd import Variable
from .rnn_reg import embedded_dropout,LockedDropout,WeightDrop
from .model import Stepper


class RNNStepper(Stepper):
    def reset(self, train=True):
        super().reset(train)
        self.m.init_hidden(train)


def seq2seq_reg(output, xtra, loss, alpha=0, beta=0):
    hs,dropped_hs = xtra
    if alpha:  # Activation Regularization
        loss = loss + sum(alpha * dropped_h.pow(2).mean() for dropped_h in dropped_hs[-1:])
    if beta:   # Temporal Activation Regularization (slowness)
        loss = loss + sum(beta * (h[1:] - h[:-1]).pow(2).mean() for h in hs[-1:])
    return loss


def reg_rnn_stepper(alpha=2, beta=1):
    reg = lambda output, xtra, loss: seq2seq_reg(output, xtra, loss, alpha, beta)
    return lambda m,opt,crit,clip: RNNStepper(m,opt,crit,clip,reg)


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)


class Seq2SeqRNN(nn.Module):
    def __init__(self, bs, ntoken, ninp, nhid, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=True):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=dropouth)
                     for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights: self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.bs,self.ninp,self.nhid,self.nlayers = bs,ninp,nhid,nlayers
        self.dropout,self.dropouti,self.dropouth,self.dropoute = dropout,dropouti,dropouth,dropoute

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        self.hidden = repackage_var(new_hidden)
        return result, raw_outputs, outputs

    def one_hidden(self, l, weight, train):
        return Variable(weight.new(1, self.bs, self.nhid if l != self.nlayers - 1 else self.ninp).zero_(), volatile=not train)

    def init_hidden(self, train):
        weight = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l, weight, train), self.one_hidden(l, weight, train))
                       for l in range(self.nlayers)]

