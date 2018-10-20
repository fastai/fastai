"NLP data loading pipeline. Supports csv, folders, and preprocessed data."
from ..torch_core import *
from .transform import *
from ..basic_data import *

__all__ = ['BaseTextDataset', 'LanguageModelLoader', 'SortSampler', 'SortishSampler', 'TextDataset', 'TextMtd',
           'pad_collate', 'read_classes', 'TextDataBunch', 'TextLMDataBunch', 'TextClasDataBunch']

TextMtd = IntEnum('TextMtd', 'DF CSV TOK IDS')

def read_classes(fname):
    with open(fname, 'r') as f:
        return [l[:-1] for l in f.readlines()]

class BaseTextDataset():
    "To directly create a text datasets from `ids` and `labels`."
    def __init__(self, ids:Collection[Collection[int]], labels:Collection[Union[int,float]], vocab_size:int, 
                 classes:Classes=None):
        self.ids,self.labels,self.vocab_size,self.classes = ids,labels,vocab_size,classes
    
    def __getitem__(self, idx:int) -> Tuple[Collection[int],Union[int,float]]: return self.ids[idx],self.labels[idx]
    def __len__(self) -> int: return len(self.ids)
                                      
class TextDataset(BaseTextDataset):
    "Basic dataset for NLP tasks."

    def __init__(self, path:PathOrStr, tokenizer:Tokenizer=None, vocab:Vocab=None, max_vocab:int=60000, chunksize:int=10000,
                 name:str='train', df=None,  min_freq:int=2, n_labels:int=1, txt_cols=None, label_cols=None,
                 create_mtd:TextMtd=TextMtd.CSV, classes:Classes=None, clear_cache:bool=False):
        self.tokenizer = ifnone(tokenizer, Tokenizer())
        self.path,self.max_vocab,self.min_freq = Path(path)/'tmp',max_vocab,min_freq
        self.label_cols = ifnone(label_cols, list(range(n_labels)))
        self.txt_cols,self.chunksize,self.name,self.df,self.create_mtd = txt_cols,chunksize,name,df,create_mtd
        self.vocab=vocab
        os.makedirs(self.path, exist_ok=True)
        if clear_cache: self.clear()
        if not self.check_toks(): self.tokenize()
        if not self.check_ids():  self.numericalize()

        if self.vocab is None: self.vocab = Vocab(self.path)
        self.ids = np.load(self.path/f'{self.name}_ids.npy')
        if os.path.isfile(self.path/f'{self.name}_lbl.npy'):
            self.labels = np.load(self.path/f'{self.name}_lbl.npy')
        else: self.labels = np.zeros((len(self.ids),), dtype=np.float32 if len(self.label_cols) > 1 else np.int64)
        self.loss_func = F.cross_entropy if len(self.label_cols) == 1 else F.binary_cross_entropy_with_logits
        if classes: self.classes = classes
        elif os.path.isfile(self.path/'classes.txt'): self.classes = read_classes(self.path/'classes.txt')
        else: self.classes = np.unique(self.labels)
        super().__init__(self.ids, self.labels, len(self.vocab.itos), self.classes)

    def general_check(self, pre_files:Collection[PathOrStr], post_files:Collection[PathOrStr]):
        "Check that post_files exist and were modified after all the prefiles."
        if not np.all([os.path.isfile(fname) for fname in post_files]): return False
        for pre_file in pre_files:
            if pre_file is None: return True
            if os.path.getmtime(pre_file) > os.path.getmtime(post_files[0]): return False
        return True

    def check_ids(self) -> bool:
        "Check if a new numericalization is needed."
        if self.create_mtd >= TextMtd.IDS: return True
        if not self.general_check([self.tok_files[0],self.id_files[1]], self.id_files): return False
        itos = pickle.load(open(self.id_files[1], 'rb'))
        h = hashlib.sha1(np.array(itos))
        with open(self.id_files[2]) as f:
            if h.hexdigest() != f.read() or len(itos) > self.max_vocab + 2: return False
        toks,ids = np.load(self.tok_files[0]),np.load(self.id_files[0])
        if len(toks) != len(ids): return False
        return True

    def check_toks(self) -> bool:
        "Check if a new tokenization is needed."
        if self.create_mtd >= TextMtd.TOK: return True
        if not self.general_check([self.csv_file], self.tok_files): return False
        with open(self.tok_files[1]) as f:
            if repr(self.tokenizer) != f.read(): return False
        return True

    def tokenize(self):
        "Tokenize the texts in the csv file."
        print(f'Tokenizing {self.name}.')
        curr_len = get_chunk_length(self.df) if (self.create_mtd == TextMtd.DF) else get_chunk_length(self.csv_file, self.chunksize)
        if (self.create_mtd == TextMtd.DF): dfs = self.df
        else: dfs = pd.read_csv(self.csv_file, header='infer' if self.txt_cols is not None else None, chunksize=self.chunksize)
        tokens,labels = [],[]
        for _ in progress_bar(range(curr_len), leave=False):
            df = next(dfs) if (type(dfs) == pd.io.parsers.TextFileReader) else self.df
            lbl_type = np.float32 if len(self.label_cols) > 1 else np.int64
            lbls = df[self.label_cols].values.astype(lbl_type) if (len(self.label_cols) > 0) else []
            self.txt_cols = ifnone(self.txt_cols, list(range(len(self.label_cols),len(df.columns))))
            texts = f'{FLD} {1} ' + df[self.txt_cols[0]].astype(str)
            for i, col in enumerate(self.txt_cols[1:]):  texts += f' {FLD} {i+2} ' + df[col].astype(str)
            toks = self.tokenizer.process_all(texts)
            tokens += toks
            labels += list(np.squeeze(lbls))
        np.save(self.tok_files[0], np.array(tokens))
        if (len(labels) > 0): np.save(self.path/f'{self.name}_lbl.npy', np.array(labels))
        with open(self.tok_files[1],'w') as f: f.write(repr(self.tokenizer))

    def numericalize(self):
        "Numericalize the tokens in the token file."
        print(f'Numericalizing {self.name}.')
        toks = np.load(self.tok_files[0])
        if self.vocab is None: self.vocab = Vocab.create(self.path, toks, self.max_vocab, self.min_freq)
        ids = np.array([self.vocab.numericalize(t) for t in toks])
        np.save(self.id_files[0], ids)

    def clear(self):
        "Remove all temporary files."
        files = [self.path/f'{self.name}_{suff}.npy' for suff in ['ids','tok','lbl']]
        if (self.create_mtd == TextMtd.DF): files.append(self.path/f'{self.name}.csv')
        for file in files:
            if os.path.isfile(file): os.remove(file)

    @property
    def csv_file(self) -> Path: return None if (self.create_mtd == TextMtd.DF) else  self.path/f'{self.name}.csv'
    @property
    def tok_files(self) -> List[Path]: return [self.path/f'{self.name}_tok.npy', self.path/'tokenize.log']
    @property
    def id_files(self) -> List[Path]:
        return [self.path/f'{self.name}_ids.npy', self.path/'itos.pkl', self.path/'numericalize.log']

    @classmethod
    def from_ids(cls, folder:PathOrStr, name:str='train', id_suff:str='_ids', lbl_suff:str='_lbl',
                 itos:str='itos.pkl', **kwargs) -> 'TextDataset':
        "Create a dataset from an id, a dictionary and label file."
        if not os.path.isfile(Path(folder)/f'{name}{lbl_suff}.npy'):
            toks = np.load(Path(folder)/f'{name}{id_suff}.npy')
            np.save(Path(folder)/f'{name}{lbl_suff}.npy', np.array([0] * len(toks)))
        orig = [Path(folder/file) for file in [f'{name}{id_suff}.npy', f'{name}{lbl_suff}.npy', itos]]
        dest = [Path(folder)/'tmp'/file for file in [f'{name}_ids.npy', f'{name}_lbl.npy', 'itos.pkl']]
        maybe_copy(orig, dest)
        return cls(folder, None, name=name, create_mtd=TextMtd.IDS, **kwargs)

    @classmethod
    def from_tokens(cls, folder:PathOrStr, name:str='train', tok_suff:str='_tok', lbl_suff:str='_lbl',
                    **kwargs) -> 'TextDataset':
        "Create a dataset from a token and label file."
        if not os.path.isfile(Path(folder)/f'{name}{lbl_suff}.npy'):
            toks = np.load(Path(folder)/f'{name}{tok_suff}.npy')
            np.save(Path(folder)/f'{name}{lbl_suff}.npy', np.array([0] * len(toks)))
        orig = [Path(folder/file) for file in [f'{name}{tok_suff}.npy', f'{name}{lbl_suff}.npy']]
        dest = [Path(folder)/'tmp'/file for file in [f'{name}_tok.npy', f'{name}_lbl.npy']]
        maybe_copy(orig, dest)
        return cls(folder, None, name=name, create_mtd=TextMtd.TOK, **kwargs)

    @classmethod
    def from_df(cls, folder:PathOrStr, df:Union[DataFrame, pd.io.parsers.TextFileReader],
                    tokenizer:Tokenizer=None, name:str='train', **kwargs) -> 'TextDataset':
        "Create a dataset from texts in a dataframe"
        tokenizer = ifnone(tokenizer, Tokenizer())
        chunksize = 1 if (type(df) == DataFrame) else df.chunksize
        return cls(folder, tokenizer, df=df, create_mtd=TextMtd.DF, name=name, chunksize=chunksize, **kwargs)

    @classmethod
    def from_csv(cls, folder:PathOrStr, tokenizer:Tokenizer=None, name:str='train', **kwargs) -> 'TextDataset':
        "Create a dataset from texts in a csv file."
        tokenizer = ifnone(tokenizer, Tokenizer())
        orig = [Path(folder)/f'{name}.csv']
        dest = [Path(folder)/'tmp'/f'{name}.csv']
        maybe_copy(orig, dest)
        return cls(folder, tokenizer, name=name, **kwargs)

    @classmethod
    def from_one_folder(cls, folder:PathOrStr, classes:Classes, tokenizer:Tokenizer=None, name:str='train',
                    shuffle:bool=True, **kwargs) -> 'TextDataset':
        "Create a dataset from one folder, labelled `classes[0]` (used for the test set)."
        tokenizer = ifnone(tokenizer, Tokenizer())
        path = Path(folder)/'tmp'
        os.makedirs(path, exist_ok=True)
        texts = []
        for fname in (Path(folder)/name).glob('*.*'):
            texts.append(fname.open('r', encoding='utf8').read())
        texts,labels = np.array(texts),np.array([classes[0]] * len(texts))
        if shuffle:
            idx = np.random.permutation(len(texts))
            texts = texts[idx]
        df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
        if os.path.isfile(path/f'{name}.csv'):
            if get_total_length(path/f'{name}.csv', 10000) != len(df):
                df.to_csv(path/f'{name}.csv', index=False, header=False)
        else: df.to_csv(path/f'{name}.csv', index=False, header=False)
        return cls(folder, tokenizer, name=name, classes=classes, **kwargs)

    @classmethod
    def from_folder(cls, folder:PathOrStr, tokenizer:Tokenizer=None, name:str='train', classes:Classes=None,
                    shuffle:bool=True, **kwargs) -> 'TextDataset':
        "Create a dataset from a folder."
        tokenizer = ifnone(tokenizer, Tokenizer())
        path = Path(folder)/'tmp'
        os.makedirs(path, exist_ok=True)
        if classes is None: classes = [cls.name for cls in find_classes(Path(folder)/name)]
        (path/'classes.txt').open('w').writelines(f'{o}\n' for o in classes)
        texts,labels = [],[]
        for idx,label in enumerate(classes):
            for fname in (Path(folder)/name/label).glob('*.*'):
                texts.append(fname.open('r', encoding='utf8').read())
                labels.append(idx)
        texts,labels = np.array(texts),np.array(labels)
        if shuffle:
            idx = np.random.permutation(len(texts))
            texts,labels = texts[idx],labels[idx]
        df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
        if os.path.isfile(path/f'{name}.csv'):
            if get_total_length(path/f'{name}.csv', 10000) != len(df):
                df.to_csv(path/f'{name}.csv', index=False, header=False)
        else: df.to_csv(path/f'{name}.csv', index=False, header=False)
        return cls(folder, tokenizer, name=name, classes=classes, **kwargs)

class LanguageModelLoader():
    "Create a dataloader with bptt slightly changing."
    def __init__(self, dataset:TextDataset, bs:int=64, bptt:int=70, backwards:bool=False):
        self.dataset,self.bs,self.bptt,self.backwards = dataset,bs,bptt,backwards
        self.data = self.batchify(np.concatenate(dataset.ids))
        self.first,self.i,self.iter = True,0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.first and self.i == 0: self.first,seq_len = False,self.bptt + 25
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self) -> int: return (self.n-1) // self.bptt

    def batchify(self, data:np.ndarray) -> LongTensor:
        "Split the corpus `data` in batches."
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs]).reshape(self.bs, -1).T
        if self.backwards: data=data[::-1].copy()
        return LongTensor(data)

    def get_batch(self, i:int, seq_len:int) -> Tuple[LongTensor, LongTensor]:
        "Create a batch at `i` of a given `seq_len`."
        seq_len = min(seq_len, len(self.data) - 1 - i)
        return self.data[i:i+seq_len], self.data[i+1:i+1+seq_len].contiguous().view(-1)

class SortSampler(Sampler):
    "Go through the text data by order of length."

    def __init__(self, data_source:NPArrayList, key:KeyFunc): self.data_source,self.key = data_source,key
    def __len__(self) -> int: return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range_of(self.data_source), key=self.key, reverse=True))


class SortishSampler(Sampler):
    "Go through the text data by order of length with a bit of randomness."

    def __init__(self, data_source:NPArrayList, key:KeyFunc, bs:int):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

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

def pad_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding."
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples): 
        if pad_first: res[-len(s[0]):,i] = LongTensor(s[0])
        else:         res[:len(s[0]):,i] = LongTensor(s[0])
    return res, torch.tensor([s[1] for s in samples]).squeeze()

class TextDataBunch(DataBunch):
    """General class to get a `DataBunch` for NLP. You should use one of its subclass, `TextLMDataBunch` or 
    `TextClasDataBunch`."""
    
    @classmethod
    def from_ids(cls, path, trn_ids:Collection[Collection[int]], trn_lbls:Collection[Union[int,float]], 
                 val_ids:Collection[Collection[int]], val_lbls:Collection[Union[int,float]], vocab_size:int,
                 tst_ids:Collection[Collection[int]]=None, classes:Classes=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a dictionary."
        path=Path(path)
        train_ds = BaseTextDataset(trn_ids, trn_lbls, vocab_size, classes)
        datasets = [train_ds, BaseTextDataset(val_ids, val_lbls, vocab_size, classes)]
        if tst_ids is not None: datasets.append(BaseTextDataset(tst_ids, np.zeros(len(tst_ids)), vocab_size, classes))
        return cls.create(datasets, path, **kwargs)
    
    @classmethod
    def from_id_files(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                      itos:str='itos.pkl', **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a dictionary."
        path=Path(path)
        k_names = ['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'id_suff', 'lbl_suff', 'clear_cache']
        txt_kwargs, kwargs = extract_kwargs(k_names, kwargs)
        train_ds = TextDataset.from_ids(path, train, itos=itos, **txt_kwargs)
        datasets = [train_ds, TextDataset.from_ids(path, valid, itos=itos, **txt_kwargs)]
        if test: datasets.append(TextDataset.from_ids(path, test, itos=itos, **txt_kwargs))
        return cls.create(datasets, path, **kwargs)

    @classmethod
    def from_tokens(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                         vocab:Vocab=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from tokens and labels."
        path=Path(path)
        k_names = ['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'tok_suff', 'lbl_suff', 'clear_cache']
        txt_kwargs, kwargs = extract_kwargs(k_names, kwargs)
        train_ds = TextDataset.from_tokens(path, train, vocab=vocab, **txt_kwargs)
        datasets = [train_ds, TextDataset.from_tokens(path, valid, vocab=train_ds.vocab, **txt_kwargs)]
        if test: datasets.append(TextDataset.from_tokens(path, test, vocab=train_ds.vocab, **txt_kwargs))
        return cls.create(datasets, path, **kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrameOrChunks, valid_df:DataFrameOrChunks,
             test_df:Optional[DataFrameOrChunks]=None, tokenizer:Tokenizer=None, vocab:Vocab=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        tokenizer = ifnone(tokenizer, Tokenizer())
        path=Path(path)
        k_names = ['max_vocab', 'min_freq', 'n_labels', 'txt_cols', 'label_cols', 'clear_cache']
        txt_kwargs, kwargs = extract_kwargs(k_names, kwargs)
        train_ds = TextDataset.from_df(path, train_df, tokenizer, 'train', vocab=vocab, **txt_kwargs)
        datasets = [train_ds, TextDataset.from_df(path, valid_df, tokenizer, 'valid', vocab=train_ds.vocab, **txt_kwargs)]
        if test_df is not None: datasets.append(TextDataset.from_df(path, test_df, tokenizer, 'test', vocab=train_ds.vocab, **txt_kwargs))
        return cls.create(datasets, path, **kwargs)
    
    @classmethod
    def from_csv(cls, path:PathOrStr, tokenizer:Tokenizer=None, train:str='train', valid:str='valid', test:Optional[str]=None,
                      vocab:Vocab=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from texts in csv files."
        tokenizer = ifnone(tokenizer, Tokenizer())
        path=Path(path)
        k_names = ['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'txt_cols', 'label_cols', 'clear_cache']
        txt_kwargs, kwargs = extract_kwargs(k_names, kwargs)
        train_ds = TextDataset.from_csv(path, tokenizer, train, vocab=vocab, **txt_kwargs)
        datasets = [train_ds, TextDataset.from_csv(path, tokenizer, valid, vocab=train_ds.vocab, **txt_kwargs)]
        if test: datasets.append(TextDataset.from_csv(path, tokenizer, test, vocab=train_ds.vocab, **txt_kwargs))
        return cls.create(datasets, path, **kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr, tokenizer:Tokenizer=None, train:str='train', valid:str='valid', test:Optional[str]=None,
                         shuffle:bool=True, vocab:Vocab=None, **kwargs):
        "Create a `TextDataBunch` from text files in folders."
        tokenizer = ifnone(tokenizer, Tokenizer())
        path=Path(path)
        k_names = ['max_vocab', 'chunksize', 'min_freq', 'n_labels', 'clear_cache']
        txt_kwargs, kwargs = extract_kwargs(k_names, kwargs)
        train_ds = TextDataset.from_folder(path, tokenizer, train, shuffle=shuffle, vocab=vocab, **txt_kwargs)
        datasets = [train_ds, TextDataset.from_folder(path, tokenizer, valid, classes=train_ds.classes,
                                        shuffle=shuffle, vocab=train_ds.vocab, **txt_kwargs)]
        if test: datasets.append(TextDataset.from_one_folder(path, tokenizer=tokenizer, folder=test, classes=train_ds.classes,
                                        shuffle=shuffle, vocab=train_ds.vocab, **txt_kwargs))
        return cls.create(datasets, path, **kwargs)

class TextLMDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, datasets:Collection[TextDataset], path:PathOrStr, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        dataloaders = [LanguageModelLoader(ds, **kwargs) for ds in datasets]
        return cls(*dataloaders, path=path)

class TextClasDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, datasets:Collection[TextDataset], path:PathOrStr, **kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification."
        bs = kwargs.pop('bs') if 'bs' in kwargs else 64
        pad_idx = kwargs.pop('pad_idx') if 'pad_idx' in kwargs else 1
        pad_first = kwargs.pop('pad_first') if 'pad_first' in kwargs else True
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first)
        train_sampler = SortishSampler(datasets[0].ids, key=lambda x: len(datasets[0].ids[x]), bs=bs//2)
        train_dl = DeviceDataLoader.create(datasets[0], bs//2, sampler=train_sampler, collate_fn=collate_fn)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            sampler = SortSampler(ds.ids, key=lambda x: len(ds.ids[x]))
            dataloaders.append(DeviceDataLoader.create(ds, bs,  sampler=sampler, collate_fn=collate_fn))
        return cls(*dataloaders, path=path)