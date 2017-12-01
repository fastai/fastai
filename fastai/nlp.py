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

import spacy

re_br = re.compile('<br />')
spacy_en = spacy.load('en')
def sub_br(x): return re_br.sub("\n", x)
def spacy_tok(x): return [tok.text for tok in spacy_en.tokenizer(sub_br(x))]

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
    def __init__(self, nf, ny, w_adj=0.4, r_adj=10):
        super().__init__()
        self.w_adj,self.r_adj = w_adj,r_adj
        self.w = nn.Embedding(nf+1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        self.r = nn.Embedding(nf+1, ny)

    def forward(self, feat_idx, feat_cnt, sz):
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        x = ((w+self.w_adj)*r/self.r_adj).sum(1)
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
    p = x[np.argwhere(y==y_i)[:,0]].sum(0)+1
    q = x[np.argwhere(y!=y_i)[:,0]].sum(0)+1
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

    def get_model(self, f, **kwargs):
        m = to_gpu(f(self.trn_ds.vocab_size, self.c, **kwargs))
        m.r.weight.data = to_gpu(self.r)
        m.r.weight.requires_grad = False
        model = BasicModel(m)
        return BOW_Learner(self, model, metrics=[accuracy_thresh(0.5)], opt_fn=optim.Adam)

    def dotprod_nb_learner(self, **kwargs): return self.get_model(DotProdNB, **kwargs)
    def nb_learner(self, **kwargs): return self.get_model(SimpleNB, **kwargs)

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
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        return self

    def __len__(self): return self.n // self.bptt - 1

    def __next__(self):
        if self.i >= self.n-1 or self.iter>=len(self): raise StopIteration
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.i, seq_len)
        self.i += seq_len
        self.iter += 1
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data[:nb*self.bs]
        data = data.view(self.bs, -1).t().contiguous()
        return to_gpu(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.cross_entropy

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))


class ConcatTextDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, newline_eos=True, **kwargs):
        fields = [('text', text_field)]
        text = []
        if os.path.isdir(path): paths=glob(f'{path}/*.*')
        else: paths=[path]
        for p in paths:
            for line in open(p): text += text_field.preprocess(line)
            if newline_eos: text.append('<eos>')

        examples = [torchtext.data.Example.fromlist([text], fields)]
        super().__init__(examples, fields, **kwargs)


class ConcatTextDatasetFromDataFrames(torchtext.data.Dataset):
    def __init__(self, df, text_field, col, newline_eos=True, **kwargs):
        fields = [('text', text_field)]
        text = []

        text += text_field.preprocess(df[col].str.cat(sep=' '))
        if (newline_eos): text.append('<eos>')

        examples = [torchtext.data.Example.fromlist([text], fields)]

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_df=None, val_df=None, test_df=None, **kwargs):
        train_data = None if train_df is None else cls(train_df, **kwargs)
        val_data = None if val_df is None else cls(val_df, **kwargs)
        test_data = None if test_df is None else cls(test_df, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class LanguageModelData():
    """
    This class provides the entry point for dealing with supported NLP tasks.
    Usage:
    1.  Use one of the factory constructors (from_dataframes, from_text-files) to
        obtain an instance of the class.
    2.  Use the get_model method to return a RNN_Learner instance (a network suited
        for NLP tasks), then proceed with training.

        Example:
            >> FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
            >> md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)

            >> em_sz = 200  # size of each embedding vector
            >> nh = 500     # number of hidden activations per layer
            >> nl = 3       # number of layers

            >> opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
            >> learner = md.get_model(opt_fn, em_sz, nh, nl,
                           dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
            >> learner.reg_fn = seq2seq_reg
            >> learner.clip=0.3

            >> learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)

    """
    def __init__(self, path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs):

        """ Constructor for the LanguageModelData class. A very important thing that happens here is
            that the field's "build_vocab" method is invoked, which builds the vocabulary
            for this NLP model. Another important thing that happens in the constructor
            is that three instances of the LanguageModelLoader is constructed; one each
            for training data (self.trn_dl), validation data (self.val_dl), and the
            testing data (self.test_dl)

        Args:
            path (str): testing path
            field (Field): torchtext field object
            trn_ds (Dataset): training dataset
            val_ds (Dataset): validation dataset
            test_ds (Dataset): testing dataset
            bs (int): batch size
            bptt (int): back propagation through time
            kwargs: other argumetns
        """
        self.bs = bs
        self.path = path
        self.trn_ds = trn_ds; self.val_ds = val_ds; self.test_ds = test_ds

        field.build_vocab(self.trn_ds, **kwargs)

        self.pad_idx = field.vocab.stoi[field.pad_token]
        self.nt = len(field.vocab)

        self.trn_dl, self.val_dl, self.test_dl = [ LanguageModelLoader(ds, bs, bptt) 
                                                    for ds in (self.trn_ds, self.val_ds, self.test_ds) ]

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        """ Method returns a RNN_Learner instance, based on the given optimizer function and other arguments.

        Args:
            opt_fn (Optimizer): the torch optimizer function to use
            emb_sz (int): embedding size
            n_hid (int): number of hidden inputs
            n_layers (int): number of hidden layers
            kwargs: other arguments

        Returns:
            An instance of the RNN_Learner class.

        """
        m = get_language_model(self.bs, self.nt, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = SingleModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

    @classmethod
    def from_dataframes(cls, path, field, col, train_df, val_df, test_df=None, bs=64, bptt=70, **kwargs):
        trn_ds, val_ds, test_ds = ConcatTextDatasetFromDataFrames.splits(text_field=field, col=col,
                                    train_df=train_df, val_df=val_df, test_df=test_df)

        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)

    @classmethod
    def from_text_files(cls, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        """ Method used to instantiate a LanguageModelData object that can be used for a
            supported nlp task.

        Args:
            path (str): the absolute path in which temporary model data will be saved
            field (Field): torchtext field
            train (str): file location of the training data
            validation (str): file location of the validation data
            test (str): file location of the testing data
            bs (int): batch size to use
            bptt (int): back propagation through time parameter (a hyper-parameter for RNN models)
            kwargs: other arguments

        Returns:
            A LanguageModelData instance, which most importantly, provides us the datasets
            for training, validation, and testing

        """
        trn_ds, val_ds, test_ds = ConcatTextDataset.splits(
                                    path, text_field=field, train=train, validation=validation, test=test)

        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)


class TextDataLoader():
    def __init__(self, src, x_fld, y_fld):
        self.src,self.x_fld,self.y_fld = src,x_fld,y_fld

    def __len__(self): return len(self.src)-1

    def __iter__(self):
        it = iter(self.src)
        for i in range(len(self)):
            b = next(it)
            yield getattr(b, self.x_fld), getattr(b, self.y_fld)


class TextModel(BasicModel):
    def get_layer_groups(self):
        return [self.model[0].encoder, self.model[0].rnns, self.model[1]]


class TextData(ModelData):
    def create_td(self, it): return TextDataLoader(it, self.text_fld, self.label_fld)

    @classmethod
    def from_splits(cls, path, splits, bs, text_name='text', label_name='label'):
        text_fld = splits[0].fields[text_name]
        label_fld = splits[0].fields[label_name]
        label_fld.build_vocab(splits[0])
        iters = torchtext.data.BucketIterator.splits(splits, batch_size=bs)
        trn_iter,val_iter,test_iter = iters[0],iters[1],None
        test_dl = None
        if len(iters) == 3:
            test_iter = iters[2]
            test_dl = TextDataLoader(test_iter, text_name, label_name)
        trn_dl = TextDataLoader(trn_iter, text_name, label_name)
        val_dl = TextDataLoader(val_iter, text_name, label_name)
        obj = cls.from_dls(path, trn_dl, val_dl, test_dl)
        obj.bs = bs
        obj.pad_idx = text_fld.vocab.stoi[text_fld.pad_token]
        obj.nt = len(text_fld.vocab)
        obj.c = len(label_fld.vocab)
        return obj

    def get_model(self, opt_fn, max_sl, bptt, emb_sz, n_hid, n_layers, **kwargs):
        m = get_rnn_classifer(max_sl, bptt, self.bs, self.c, self.nt, emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers,
                              pad_token=self.pad_idx, **kwargs)
        model = TextModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

