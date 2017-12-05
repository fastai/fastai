from .torch_imports import *
from .core import *
from functools import wraps
from torch.autograd import Variable


def dropout_mask(x, sz, dropout):
    """ Method returns a new tensor whose input is the shape determined by sz.

    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide to keep the activations.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).

    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.autograd.Variable(torch.Tensor(2, 3, 4).uniform_(0, 1), requires_grad=False)
        >>> a_Var

            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217

            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]

        >>> a_mask = dropout_mask(a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask

            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new(*sz).bernoulli_(1-dropout)/(1-dropout)


class LockedDropout(nn.Module):
    """A custom module that applies dropout to the activations passing through it.

    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        """Applies the dropout to input so that approximately specified
        fraction of the activation is dropped (or zero-ed) out. The original
        input is also rescaled because of how dropout_mask works.

        Args:
            x (autograd.Variable): torch Variable

        Returns:
            autograd.Variable

        """
        if not self.training or not self.p: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x


class WeightDrop(torch.nn.Module):
    """A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped layer based
    on a specified dropout.

    The dropout is applied by using torch.nn.functional.dropout()
    (see method _setweights). The main motivation for using this approach
    stems from the work done in "Regularizing and Optimizing LSTM Language Models"
    (https://arxiv.org/pdf/1708.02182.pdf)

    """
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        super().__init__()
        self.module,self.weights,self.dropout = module,weights,dropout
        self._setup()

    def _setup(self):
        """ Method processes the weights in the wrapped module by essentially deleting them
        (the weights) from the module's parameters group, then re-registering them.
        The weights are also registered as slightly different attribute names, i.e.
        appending the name with a '_raw' string.

        An important thing done is that the flatten_parameters operation is set to a
        noop (no operation) function if the wrapped module is an instance of torch.nn.RNNBase.

        I'm not sure why either of the above is executed. Very little information exists
        regarding the flatten_parameters() method. I suppose there's some issue with
        weights compaction for a RNN module in torch. If anything, sources at

        https://discuss.pytorch.org/t/when-to-call-flatten-parameters-in-lstm-when-using-multi-gpus/10219 and
        https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282

        actually seem to suggest that flatten_parameters() should be called for LSTMs,
        so that weights are compacted and the execution runs faster across multiple GPUs.

        Args:
            self (WeightDrop): the self instance

        Returns:
            None
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if isinstance(self.module, torch.nn.RNNBase): self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))


    def _setweights(self):

        """Method applies the specified dropout on the weights in the wrapped model.

        Args:
            self (WeightDrop): the self instance

        Returns:
             None
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        """ Method updates the weights on the wrapped layer based on the dropout specified
        during layer instantiation. Then the input is  forward-propagated using the
        forward method specified for the wrapped module.

        Args:
            *args: optional arguments

            As is standard for most torch modules, one of the arguments will always be an input tensor.

        Returns:
             tensor obtained after forward propagation.
        """
        self._setweights()
        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """Method applies dropout to the embedding input.

    The dropout is applied using the dropout_mask. Essentially,
    for all rows (w_1, w_2, .... w_N)  in the embedding matrix, this
    method zeros out a fraction of these elements. Then the existing
    values are rescaled using the 1/(1-dropout) value. Additional rescaling
    can be specified using the scale parameter.

    Finally, the method uses the modified (masked_embedding_weight)
    to evaluate the ouput tensor.

    Args:
        embed (nn.Embedding): torch Embedding module
        words (nn.Variable): torch Variable carrying a tensor
        dropout (float): dropout value to apply
        scale (float): additional rescaling to apply

    Returns:
        torch Variable
    """
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

