"NLP data loading pipeline. Supports csv, folders, and preprocessed data."
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *

__all__ = ['LanguageModelLoader', 'SortSampler', 'SortishSampler', 'TextBase', 'TextDataset', 'TextMtd', 'TextFileList',
           'pad_collate', 'TextDataBunch', 'TextLMDataBunch', 'TextClasDataBunch', 'SplitDatasetsText',
           'NumericalizedDataset', 'TokenizedDataset']

TextMtd = IntEnum('TextMtd', 'DF TOK IDS')
text_extensions = ['.txt']

class TextFileList(InputList):
    "A list of inputs. Contain methods to get the corresponding labels."
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=text_extensions, recurse=True)->'ImageFileList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions=extensions, recurse=recurse), path)
    
class SplitDatasetsText(SplitDatasets):
    def tokenize(self, tokenizer:Tokenizer=None, chunksize:int=10000):
        "Tokenize `self.datasets` with `tokenizer` by bits of `chunksize`."
        self.datasets = [ds.tokenize(tokenizer, chunksize) for ds in self.datasets]
        return self
        
    def numericalize(self, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=2):
        "Numericalize `self.datasets` with `vocab` or by creating one on the training set with `max_vocab` and `min_freq`."
        dss = self.datasets
        train_ds = dss[0].numericalize(vocab, max_vocab, min_freq)
        self.datasets = [train_ds] + [ds.numericalize(train_ds.vocab) for ds in dss[1:]]
        return self
    
    def databunch(self, cls_func, path:PathOrStr=None, **kwargs):
        "Create an `cls_func` from self, `path` will override `self.path`, `kwargs` are passed to `cls_func.create`."
        path = Path(ifnone(path, self.path))
        return cls_func.create(*self.datasets, path=path, **kwargs) 
    
class TextBase(LabelDataset):
    __splits_class__ = SplitDatasetsText
    def __init__(self, x:Collection[Any], labels:Collection[Union[int,float]]=None, classes:Collection[Any]=None, 
                 encode_classes:bool=True):
        if classes is None: classes = uniqueify(labels)
        super().__init__(classes=classes)
        self.x = np.array(x)
        if labels is None: self.y = np.zeros(len(x)) 
        elif encode_classes and len(labels.shape) == 1: self.y = np.array([self.class2idx[o] for o in labels], dtype=np.int64) 
        else: self.y = labels
        
class NumericalizedDataset(TextBase):
    "To directly create a text dataset from `ids` and `labels`."
    def __init__(self, vocab:Vocab, ids:Collection[Collection[int]], labels:Collection[Union[int,float]]=None,
                 classes:Collection[Any]=None, encode_classes:bool=True):
        super().__init__(ids, labels, classes, encode_classes)
        self.vocab, self.vocab_size = vocab, len(vocab.itos)
        self.loss_func = F.cross_entropy if len(self.y.shape) <= 1 else F.binary_cross_entropy_with_logits
    
    def get_text_item(self, idx, sep=' ', max_len:int=None):
        "Return the text in `idx`, tokens separated by `sep` and cutting at `max_len`."
        inp = self.x[idx] if max_len is None else self.x[idx][:max_len]
        if isinstance(self.y[idx], Iterable): title = ';'.join([self.classes[i] for i,v in enumerate(self.y[idx]) if v == 1.])
        else: title = self.classes[self.y[idx]]
        return self.vocab.textify(inp, sep), title
    
    def save(self, path:Path, name:str):
        "Save the dataset in `path` with `name`."
        os.makedirs(path, exist_ok=True)
        np.save(path/f'{name}_ids.npy', self.x)
        np.save(path/f'{name}_lbl.npy', self.y)
        pickle.dump(self.vocab.itos, open(path/'itos.pkl', 'wb'))
        save_texts(path/'classes.txt', self.classes)
        
    @classmethod
    def load(cls, path:Path, name:str):
        "Load a `NumericalizedDataset` from `path` in `name`."
        vocab = Vocab(pickle.load(open(path/f'itos.pkl', 'rb')))
        x,y = np.load(path/f'{name}_ids.npy'), np.load(path/f'{name}_lbl.npy')
        classes = loadtxt_str(path/'classes.txt')
        return cls(vocab, x, y, classes, encode_classes=False)
        
class TokenizedDataset(TextBase):
    "To create a text dataset from `tokens` and `labels`."
    def __init__(self, tokens:Collection[Collection[str]], labels:Collection[Union[int,float]]=None, 
                 classes:Collection[Any]=None, encode_classes:bool=True):
        super().__init__(tokens, labels, classes, encode_classes)
        
    def save(self, path:Path, name:str):
        "Save the dataset in `path` with `name`."
        os.makedirs(path, exist_ok=True)
        np.save(path/f'name_tok.npy', self.x)
        np.save(path/f'name_lbl.npy', self.y)
        np.savetxt(path/'classes.txt', self.classes.as_type(str))
    
    def numericalize(self, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=2)->'NumericalizedDataset':
        "Numericalize the tokens with `vocab` (if not None) otherwise create one with `max_vocab` and `min_freq` from tokens."
        vocab = ifnone(vocab, Vocab.create(self.x, max_vocab, min_freq))
        ids = np.array([vocab.numericalize(t) for t in self.x])
        return NumericalizedDataset(vocab, ids, self.y, self.classes, encode_classes=False)

def _join_texts(texts:Collection[str], mark_fields:bool=True):
    if len(texts.shape) == 1: texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{FLD} {1} ' + df[0] if mark_fields else df[txt_cols[0]]
    for i in range(1,len(df.columns)):  
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i]
    return text_col.values
    
class TextDataset(TextBase):
    "Basic dataset for NLP tasks."
    def __init__(self, texts:Collection[str], labels:Collection[Any]=None, classes:Collection[Any]=None,
                 mark_fields:bool=True, encode_classes:bool=True, is_fnames:bool=False):
        if is_fnames:
            fnames,texts = texts.copy(),[]
            for f in fnames:
                with open(f,'r') as f: texts.append(''.join(f.readlines()))
        texts = _join_texts(np.array(texts), mark_fields)
        super().__init__(texts, labels, classes, encode_classes)

    @classmethod
    def from_df(cls, df:DataFrame, classes:Collection[Any]=None, n_labels:int=1, txt_cols:Collection[Union[int,str]]=None, 
                label_cols:Collection[Union[int,str]]=None, mark_fields:bool=True) -> 'TextDataset':
        "Create a `TextDataset` from the texts in a dataframe"
        label_cols = ifnone(label_cols, list(range(n_labels)))
        if classes is None:
            if len(label_cols) == 0:   classes = [0]
            elif len(label_cols) == 1: classes = df.iloc[:,df_names_to_idx(label_cols, df)[0]].unique()
            else:                      classes = label_cols
        dtype = np.int64 if len(label_cols) <= 1 else np.float32
        labels = np.squeeze(df.iloc[:,df_names_to_idx(label_cols, df)].astype(dtype).values)
        txt_cols = ifnone(txt_cols, list(range(len(label_cols),len(df.columns))))
        texts = np.squeeze(df.iloc[:,df_names_to_idx(txt_cols, df)].astype(str).values)
        return cls(texts, labels, classes, mark_fields)

    @staticmethod
    def _folder_files(folder:Path, label:str, extensions:Collection[str]=text_extensions)->Tuple[str,str]:
        "From `folder` return texts in files and labels. The labels are all `label`."
        fnames = get_files(folder, extensions='.txt')
        texts = []
        for f in fnames:
            with open(f,'r') as f: texts.append(f.readlines())
        return texts,[label]*len(texts)
    
    @classmethod
    def from_folder(cls, path:PathOrStr, classes:Collection[Any]=None, valid_pct:float=0.,
                    extensions:Collection[str]=text_extensions, mark_fields:bool=True) -> 'TextDataset':
        """Create a `TextDataset` by scanning the subfolders in `path` for files with `extensions`. 
        Only keep those with labels in `classes`. If `valid_pct` is not 0., splits the data randomly in two datasets accordingly.
        `mark_fields` is passed to the initialization. """
        path = Path(path)
        classes = ifnone(classes, [cls.name for cls in find_classes(path)])
        texts, labels, keep = [], [], {}
        for cl in classes:
            t,l = cls._folder_files(path/cl, cl, extensions=extensions)
            texts+=t; labels+=l
            keep[cl] = len(t)
        classes = [cl for cl in classes if keep[cl]]
        if valid_pct == 0.: return cls(texts, labels, classes, mark_fields)
        return [cls(*a, classes, mark_fields) for a in random_split(valid_pct, texts, labels)]
    
    @classmethod
    def from_one_folder(cls, path:PathOrStr, classes:Collection[Any], extensions:Collection[str]=text_extensions,  
                        mark_fields:bool=True) -> 'TextDataset':
        """Create a `TextDataset` by scanning the subfolders in `path` for files with `extensions`. 
        Label all of them with `classes[0]`.  `mark_fields` is passed to the initialization. """
        path = Path(path)
        text,labels = self._folder_files(path, classes[0], extensions=extensions)
        return cls(texts, labels, classes, mark_fields)
    
    def tokenize(self, tokenizer:Tokenizer=None, chunksize:int=10000)->'TokenizedDataset':
        "Tokenize the texts with `tokenizer` by bits of `chunksize`."
        tokenizer = ifnone(tokenizer, Tokenizer())
        tokens = []
        for i in progress_bar(range(0,len(self.x),chunksize), leave=False):
            tokens += tokenizer.process_all(self.x[i:i+chunksize])
        return TokenizedDataset(tokens, self.y, self.classes, encode_classes=False)

class LanguageModelLoader():
    "Create a dataloader with bptt slightly changing."
    def __init__(self, dataset:TextDataset, bs:int=64, bptt:int=70, backwards:bool=False, shuffle:bool=False):
        self.dataset,self.bs,self.bptt,self.backwards,self.shuffle = dataset,bs,bptt,backwards,shuffle
        self.first,self.i,self.iter = True,0,0
        self.n = len(np.concatenate(dataset.x)) // self.bs
        self.num_workers = 0

    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None:
            yield LongTensor(getattr(self.dataset, 'item')).unsqueeze(1),LongTensor([0])
        idx = np.random.permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        self.data = self.batchify(np.concatenate([self.dataset.x[i] for i in idx]))
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
    return res, tensor([s[1] for s in samples])

def _parse_kwargs(kwargs):
    txt_kwargs, kwargs = extract_kwargs(['n_labels', 'txt_cols', 'label_cols'], kwargs)
    tok_kwargs, kwargs = extract_kwargs(['chunksize'], kwargs)
    num_kwargs, kwargs = extract_kwargs(['max_vocab', 'min_freq'], kwargs)
    return txt_kwargs, tok_kwargs, num_kwargs, kwargs

class TextDataBunch(DataBunch):
    """General class to get a `DataBunch` for NLP. You should use one of its subclass, `TextLMDataBunch` or
    `TextClasDataBunch`."""
    
    def save(self, cache_name:PathOrStr='tmp'):
        "Save the `DataBunch` in `self.path/cache_name` folder."
        os.makedirs(self.path/cache_name, exist_ok=True)
        cache_path = self.path/cache_name
        pickle.dump(self.train_ds.vocab.itos, open(cache_path/f'itos.pkl', 'wb'))
        np.save(cache_path/f'train_ids.npy', self.train_ds.x)
        np.save(cache_path/f'train_lbl.npy', self.train_ds.y)
        np.save(cache_path/f'valid_ids.npy', self.valid_ds.x)
        np.save(cache_path/f'valid_lbl.npy', self.valid_ds.y)
        if self.test_dl is not None: np.save(cache_path/f'test_ids.npy', self.test_ds.x)
        save_texts(cache_path/'classes.txt', self.train_ds.classes)
    
    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, trn_ids:Collection[Collection[int]], val_ids:Collection[Collection[int]], 
                 tst_ids:Collection[Collection[int]]=None, trn_lbls:Collection[Union[int,float]]=None, 
                 val_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a dictionary."
        train_ds = NumericalizedDataset(vocab, trn_ids, trn_lbls, classes, encode_classes=False)
        datasets = [train_ds, NumericalizedDataset(vocab, val_ids, val_lbls, train_ds.classes, encode_classes=False)]
        if tst_ids is not None: datasets.append(NumericalizedDataset(vocab, tst_ids, None, train_ds.classes, encode_classes=False))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def load(cls, path:PathOrStr, cache_name:PathOrStr='tmp', **kwargs):
        "Load a `TextDataBunch` from `path/cache_name`. `kwargs` are passed to the dataloader creation."
        cache_path = Path(path)/cache_name
        vocab = Vocab(pickle.load(open(cache_path/f'itos.pkl', 'rb')))
        trn_ids,trn_lbls = np.load(cache_path/f'train_ids.npy'), np.load(cache_path/f'train_lbl.npy')
        val_ids,val_lbls = np.load(cache_path/f'valid_ids.npy'), np.load(cache_path/f'valid_lbl.npy')
        tst_ids = np.load(cache_path/f'test_ids.npy') if os.path.isfile(cache_path/f'test_ids.npy') else None
        classes = loadtxt_str(cache_path/'classes.txt')
        return cls.from_ids(path, vocab, trn_ids, val_ids, tst_ids, trn_lbls, val_lbls, classes, **kwargs)

    @classmethod
    def from_tokens(cls, path:PathOrStr, trn_tok:Collection[Collection[str]], trn_lbls:Collection[Union[int,float]],
                 val_tok:Collection[Collection[str]], val_lbls:Collection[Union[int,float]], vocab:Vocab=None, 
                 tst_tok:Collection[Collection[str]]=None, classes:Collection[Any]=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from tokens and labels."
        num_kwargs, kwargs = extract_kwargs(['max_vocab', 'min_freq'], kwargs)
        train_ds = TokenizedDataset(trn_tok, trn_lbls, classes).numericalize(vocab, **num_kwargs)
        datasets = [train_ds, TokenizedDataset(val_tok, val_lbls, train_ds.classes).numericalize(vocab)]
        if test: datasets.append(TokenizedDataset(tst_tok, [0]*len(tst_tok), train_ds.classes).numericalize(vocab))
        return cls.create(*datasets, path=path, **kwargs)
    
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None, 
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        txt_kwargs, tok_kwargs, num_kwargs, kwargs = _parse_kwargs(kwargs)
        datasets = [(TextDataset.from_df(train_df, classes, **txt_kwargs)
                    .tokenize(tokenizer, **tok_kwargs)
                    .numericalize(vocab, **num_kwargs))]
        dfs = [valid_df] if test_df is None else [valid_df, test_df]
        for df in dfs:
            datasets.append((TextDataset.from_df(df, datasets[0].classes, **txt_kwargs)
                    .tokenize(tokenizer, **tok_kwargs)
                    .numericalize(datasets[0].vocab, **num_kwargs)))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name, valid_pct:float=0.2, test:Optional[str]=None,
                 tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, header = 'infer', **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from texts in csv files."
        df = pd.read_csv(Path(path)/csv_name, header=header)
        idx = np.random.permutation(len(df))
        cut = int(valid_pct * len(df)) + 1
        train_df, valid_df = df[cut:], df[:cut]
        test_df = None if test is None else pd.read_csv(Path(path)/test, header=header)
        return cls.from_df(path, train_df, valid_df, test_df, tokenizer, vocab, classes, **kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                    tokenizer:Tokenizer=None, vocab:Vocab=None, **kwargs):
        "Create a `TextDataBunch` from text files in folders."
        txt_kwargs, tok_kwargs, num_kwargs, kwargs = _parse_kwargs(kwargs)
        train_ds = (TextDataset.from_folder(train, classes, **txt_kwargs)
                    .tokenize(tokenizer, **tok_kwargs)
                    .numericalize(vocab, **num_kwargs))
        datasets = [train_ds, (TextDataset.from_folder(valid, train_ds.classes, **txt_kwargs)
                               .tokenize(tokenizer, **tok_kwargs)
                               .numericalize(train_ds.vocab, **num_kwargs))]
        if test:
            datasets.append((TextDataset.from_one_folder(valid, train_ds.classes, **txt_kwargs)
                             .tokenize(tokenizer, **tok_kwargs)
                             .numericalize(train_ds.vocab, **num_kwargs)))
        return cls.create(*datasets, path=path, **kwargs)

def _treat_html(o:str)->str:
    return o.replace('\n','\\n')

def _text2html_table(items:Collection[Collection[str]], widths:Collection[int])->str:
    html_code = f"<table>"
    for w in widths: html_code += f"  <col width='{w}%'>"
    for line in items:
        html_code += "  <tr>\n"
        html_code += "\n".join([f"    <th>{_treat_html(o)}</th>" for o in line if len(o) >= 1])
        html_code += "\n  </tr>\n"
    return html_code + "</table>\n"

class TextLMDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        dataloaders = [LanguageModelLoader(ds, shuffle=(i==0), **kwargs) for i,ds in enumerate(datasets)]
        return cls(*dataloaders, path=path)
    
    def show_batch(self, sep=' ', ds_type:DatasetType=DatasetType.Train, rows:int=10, max_len:int=100):
        "Show `rows` texts from a batch of `ds_type`, tokens are joined with `sep`, truncated at `max_len`."
        from IPython.display import clear_output, display, HTML
        dl = self.dl(ds_type)
        x,y = next(iter(dl))
        items = [['idx','text']]
        for i in range(rows):
            inp = self.x[:,i] if max_len is None else x[:,i][:max_len]
            items.append([str(i), self.train_ds.vocab.textify(inp.cpu(), sep=sep)])
        display(HTML(_text2html_table(items, [5,95])))

class TextClasDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs=64, pad_idx=1, pad_first=True, 
               **kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0].x[t]), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs//2, sampler=train_sampler, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            sampler = SortSampler(ds.x, key=lambda t: len(ds.x[t]))
            dataloaders.append(DataLoader(ds, batch_size=bs,  sampler=sampler, **kwargs))
        return cls(*dataloaders, path=path, collate_fn=collate_fn)
    
    def show_batch(self, sep=' ', ds_type:DatasetType=DatasetType.Train, rows:int=10, max_len:int=100):
        "Show `rows` texts from a batch of `ds_type`, tokens are joined with `sep`, truncated at `max_len`."
        from IPython.display import clear_output, display, HTML
        dl = self.dl(ds_type)
        b_idx = next(iter(dl.batch_sampler))
        first = dl.get_text_item(0, sep, max_len)
        items = [['text', 'label']]
        for i in b_idx[:rows]:
            items.append(list(dl.get_text_item(i, sep, max_len)))
        display(HTML(_text2html_table(items, [90,10])))