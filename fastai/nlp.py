from .imports import *
from .torch_imports import *
from .core import *
from .model import *
from .dataset import *
from .learner import *
from .lm_rnn import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torchtext.datasets import language_modeling


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def texts_from_files(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        t = [o.strip() for o in open(path, encoding = "ISO-8859-1")]
        texts += t
        labels += ([idx] * len(t))
    return texts,np.array(labels)

def texts_from_folders(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(open(fpath).read())
            labels.append(idx)
    return texts,np.array(labels)

class DotProdNB(nn.Module):
    def __init__(self, nf, ny):
        super().__init__()
        self.w = nn.Embedding(nf+1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        self.r = nn.Embedding(nf+1, ny)

    def forward(self, feat_idx, feat_cnt, sz):
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        x = ((w+0.4)*r/10).sum(1)
        return F.softmax(x)

class SimpleNB(nn.Module):
    def __init__(self, nf, ny):
        super().__init__()
        self.r = nn.Embedding(nf+1, ny, padding_idx=0)
        self.b = nn.Parameter(torch.zeros(ny,))

    def forward(self, feat_idx, feat_cnt, sz):
        r = self.r(feat_idx)
        x = r.sum(1)+self.b
        return F.softmax(x)

class BOW_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.l1_loss

def calc_r(y_i, x, y):
    p = x[y==y_i].sum(0)+1
    q = x[y!=y_i].sum(0)+1
    return np.log((p/p.sum())/(q/q.sum()))

class BOW_Dataset(Dataset):
    def __init__(self, bow, y, max_len): 
        self.bow,self.max_len = bow,max_len
        self.c = int(y.max())+1
        self.n,self.vocab_size = bow.shape
        self.y = one_hot(y,self.c)
        x = self.bow.sign()
        self.r = np.stack([calc_r(i, x, y).A1 for i in range(self.c)]).T

    def do_pad(self, prepend, a):
        return np.array((prepend+a.tolist())[-self.max_len:])

    def pad_row(self, row):
        prepend = [0] * max(self.max_len - len(row.indices), 0)
        return self.do_pad(prepend, row.indices+1), self.do_pad(prepend, row.data)

    def __getitem__(self,i):
        row = self.bow.getrow(i)
        ind,data = self.pad_row(row)
        return ind, data, len(row.indices), self.y[i].astype(np.float32)

    def __len__(self): return len(self.bow.indptr)-1


class TextClassifierData(ModelData):
    @property
    def c(self): return self.trn_ds.c

    @property
    def r(self):
        return torch.Tensor(np.concatenate([np.zeros((1,self.c)), self.trn_ds.r]))

    def get_model(self, f):
        m = f(self.trn_ds.vocab_size, self.c).cuda()
        m.r.weight.data = self.r.cuda()
        m.r.weight.requires_grad = False
        model = BasicModel(m)
        return BOW_Learner(self, model, metrics=[accuracy_thresh(0.5)], opt_fn=optim.Adam)

    def dotprod_nb_learner(self): return self.get_model(DotProdNB)
    def nb_learner(self): return self.get_model(SimpleNB)

    @classmethod
    def from_bow(cls, trn_bow, trn_y, val_bow, val_y, sl):
        trn_ds = BOW_Dataset(trn_bow, trn_y, sl)
        val_ds = BOW_Dataset(val_bow, val_y, sl)
        trn_dl = DataLoader(trn_ds, 64, True)
        val_dl = DataLoader(val_ds, 64, False)
        return cls('.', trn_dl, val_dl)


class LanguageModelLoader():

    def __init__(self, ds, bs, bptt):
        self.bs,self.bptt = bs,bptt
        text = sum([o.text for o in ds], [])
        fld = ds.fields['text']
        nums = fld.numericalize([text])
        self.data = self.batchify(nums)
        self.i = 0
        self.n = len(self.data)

    def __iter__(self):
        self.i=0
        return self

    def __len__(self): return self.n // self.bptt

    def __next__(self):
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.data, self.i, seq_len)
        self.i += seq_len
        if self.i > self.n-2: self.i=0
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data.narrow(0, 0, nb * self.bs)
        data = data.view(self.bs, -1).t().contiguous()
        return data.cuda()

    def get_batch(self, source, i, seq_len):
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class LanguageModelLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.cross_entropy


class LanguageModelData():
    def __init__(self, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        self.path = path
        self.trn_ds,self.val_ds,self.test_ds = language_modeling.LanguageModelingDataset.splits(
            path, text_field=field, train=train, validation=validation, test=test)
        field.build_vocab(self.trn_ds, **kwargs)
        self.nt = len(field.vocab)
        self.trn_dl,self.val_dl,self.test_dl = [LanguageModelLoader(ds, bs, bptt) for ds in
                                               (self.trn_ds,self.val_ds,self.test_ds)]

    def get_model(self, opt_fn, ninp, nhid, nlayers, **kwargs):
        m = Seq2SeqRNN(self.nt, ninp, nhid, nlayers, **kwargs).cuda()
        model = BasicModel(m)
        return LanguageModelLearner(self, model, opt_fn=opt_fn)

