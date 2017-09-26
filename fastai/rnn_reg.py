from .torch_imports import *
from .core import *
from functools import wraps
from torch.autograd import Variable


def dropout_mask(x, sz, dropout): return x.new(*sz).bernoulli_(1-dropout)/(1-dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or not self.p: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x


class WeightDrop(torch.nn.Module):
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        super().__init__()
        self.module,self.weights,self.dropout = module,weights,dropout
        self._setup()

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if isinstance(self.module, torch.nn.RNNBase): self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))


    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = Variable(dropout_mask(embed.weight.data, (embed.weight.size(0), 1), dropout))
    masked_embed_weight = mask * embed.weight
  else: masked_embed_weight = embed.weight
  if scale: masked_embed_weight = scale * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None: padding_idx = -1
  X = embed._backend.Embedding.apply(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

