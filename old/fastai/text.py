from .core import *
from .learner import *
from .lm_rnn import *
from torch.utils.data.sampler import Sampler
import spacy
from spacy.symbols import ORTH

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def texts_labels_from_folders(path, folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(idx)
    return texts, np.array(labels).astype(np.int64)

def numericalize_tok(tokens, max_vocab=50000, min_freq=0, unk_tok="_unk_", pad_tok="_pad_", bos_tok="_bos_", eos_tok="_eos_"):
    """Takes in text tokens and returns int2tok and tok2int converters

        Arguments:
        tokens(list): List of tokens. Can be a list of strings, or a list of lists of strings.
        max_vocab(int): Number of tokens to return in the vocab (sorted by frequency)
        min_freq(int): Minimum number of instances a token must be present in order to be preserved.
        unk_tok(str): Token to use when unknown tokens are encountered in the source text.
        pad_tok(str): Token to use when padding sequences.
    """
    if isinstance(tokens, str):
        raise ValueError("Expected to receive a list of tokens. Received a string instead")
    if isinstance(tokens[0], list):
        tokens = [p for o in tokens for p in o]
    freq = Counter(tokens)
    int2tok = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    unk_id = 3
    int2tok.insert(0, bos_tok)
    int2tok.insert(1, pad_tok)
    int2tok.insert(2, eos_tok)
    int2tok.insert(unk_id, unk_tok)
    tok2int = collections.defaultdict(lambda:unk_id, {v:k for k,v in enumerate(int2tok)})
    return int2tok, tok2int

class Tokenizer():
    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self,x): return self.re_br.sub("\n", x)

    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP,TOK_SENT,TOK_MIX = ' t_up ',' t_st ',' t_mx '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
    #                 else [TOK_SENT,s.lower()] if (s.istitle() and re_word.search(prev))
                    else [s.lower()])
    #         if re_nonsp.search(s): prev = s
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en', ncpus = None):
        ncpus = ncpus or num_cpus()//2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])


class TextDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x,self.y,self.backwards,self.sos,self.eos = x,y,backwards,sos,eos

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x),self.y[idx]

    def __len__(self): return len(self.x)


class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.
    """
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


class LanguageModelLoader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of the way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]


class LanguageModelData():
    def __init__(self, path, pad_idx, n_tok, trn_dl, val_dl, test_dl=None, **kwargs):
        self.path,self.pad_idx,self.n_tok = path,pad_idx,n_tok
        self.trn_dl,self.val_dl,self.test_dl = trn_dl,val_dl,test_dl

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = get_language_model(self.n_tok, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)


class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy
    def fit(self, *args, **kwargs): return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))


class TextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), (self.model[1])]

