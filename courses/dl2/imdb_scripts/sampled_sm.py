from fastai.learner import *
from fastai.text import *

def resample_vocab(itos, trn, val, sz):
    freqs = Counter(trn)
    itos2 = [o for o,p in freqs.most_common()][:sz]
    itos2.insert(0,1)
    itos2.insert(0,0)
    stoi2 = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos2)})

    trn = np.array([stoi2[o] for o in trn])
    val = np.array([stoi2[o] for o in val])

    itos3 = [itos[o] for o in itos2]
    stoi3 = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos3)})
    return trn,val,itos3,stoi3


def get_prs(c, nt):
    uni_counter = Counter(c)
    uni_counts = np.array([uni_counter[o] for o in range(nt)])
    return uni_counts/uni_counts.sum()

class LinearDecoder(nn.Module):
    initrange=0.1
    def __init__(self, n_out, nhid, dropout, tie_encoder=None, decode_train=True):
        super().__init__()
        self.decode_train = decode_train
        self.decoder = nn.Linear(nhid, n_out, bias=False)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        output = output.view(output.size(0)*output.size(1), output.size(2))
        if self.decode_train or not self.training:
            decoded = self.decoder(output)
            output = decoded.view(-1, decoded.size(1))
        return output, raw_outputs, outputs


def get_language_model(n_tok, em_sz, nhid, nlayers, pad_token, decode_train=True, dropouts=None):
    if dropouts is None: dropouts = [0.5,0.4,0.5,0.05,0.3]
    rnn_enc = RNN_Encoder(n_tok, em_sz, n_hid=nhid, n_layers=nlayers, pad_token=pad_token,
                 dropouti=dropouts[0], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
    rnn_dec = LinearDecoder(n_tok, em_sz, dropouts[1], decode_train=decode_train, tie_encoder=rnn_enc.encoder)
    return SequentialRNN(rnn_enc, rnn_dec)


def pt_sample(pr, ns):
    w = -torch.log(cuda.FloatTensor(len(pr)).uniform_())/(pr+1e-10)
    return torch.topk(w, ns, largest=False)[1]


class CrossEntDecoder(nn.Module):
    initrange=0.1
    def __init__(self, prs, decoder, n_neg=4000, sampled=True):
        super().__init__()
        self.prs,self.decoder,self.sampled = T(prs).cuda(),decoder,sampled
        self.set_n_neg(n_neg)

    def set_n_neg(self, n_neg): self.n_neg = n_neg

    def get_rand_idxs(self): return pt_sample(self.prs, self.n_neg)

    def sampled_softmax(self, input, target):
        idxs = V(self.get_rand_idxs())
        dw = self.decoder.weight
        #db = self.decoder.bias
        output = input @ dw[idxs].t() #+ db[idxs]
        max_output = output.max()
        output = output - max_output
        num = (dw[target] * input).sum(1) - max_output
        negs = torch.exp(num) + (torch.exp(output)*2).sum(1)
        return (torch.log(negs) - num).mean()

    def forward(self, input, target):
        if self.decoder.training:
            if self.sampled: return self.sampled_softmax(input, target)
            else: input = self.decoder(input)
        return F.cross_entropy(input, target)

def get_learner(drops, n_neg, sampled, md, em_sz, nh, nl, opt_fn, prs):
    m = to_gpu(get_language_model(md.n_tok, em_sz, nh, nl, md.pad_idx, decode_train=False, dropouts=drops))
    crit = CrossEntDecoder(prs, m[1].decoder, n_neg=n_neg, sampled=sampled).cuda()
    learner = RNN_Learner(md, LanguageModel(m), opt_fn=opt_fn)
    crit.dw = learner.model[0].encoder.weight
    learner.crit = crit
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    return learner,crit

