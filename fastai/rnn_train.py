import torch
import torch.nn as nn
from torch.autograd import Variable
from .model import Stepper


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)


def get_rnn_stepper(bs, alpha=2, beta=1):
    return lambda m,opt,crit,clip: RNNStepper(m,opt,crit,clip,bs,alpha,beta)


class RNNStepper(Stepper):
    def __init__(self, m, opt, crit, clip, bs, alpha, beta):
        super().__init__(m,opt,crit,clip)
        self.hidden = m.init_hidden(bs)
        self.alpha,self.beta = alpha,beta

    def step(self, xs,y):
        output,hidden,hs,dropped_hs = self.m(*xs, self.hidden, return_h=True)
        self.hidden = repackage_var(hidden)
        self.opt.zero_grad()

        raw_loss = self.crit(output.view_as(y), y)
        loss = raw_loss
        # Activation Regularization
        if self.alpha:
            loss = loss + sum(self.alpha * dropped_h.pow(2).mean() for dropped_h in dropped_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if self.beta:
            loss = loss + sum(self.beta * (h[1:] - h[:-1]).pow(2).mean() for h in hs[-1:])
        loss.backward()
        # Gradient clipping
        if self.clip:
            nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        return raw_loss.data[0]


def batchify(data, bs):
    nb = data.size(0) // bs
    data = data.narrow(0, 0, nb * bs)
    data = data.view(bs, -1).t().contiguous()
    return data.cuda()


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


def evaluate(data_source, bs=10):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(bs)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, y = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, y).data
    return total_loss[0] / len(data_source)


def train():
    ntokens = len(corpus.dictionary)
    while i < train_data.size(0) - 2:
        data, y = get_batch(train_data, i, seq_len=seq_len)

