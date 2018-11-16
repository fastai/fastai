"NLP data loading pipeline. Supports csv, folders, and preprocessed data."
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *

__all__ = ['LanguageModelLoader', 'SortSampler', 'SortishSampler', 'TextList', 'pad_collate', 'TextDataBunch',
           'TextLMDataBunch', 'TextClasDataBunch', 'Text', 'open_text', 'TokenizeProcessor', 'NumericalizeProcessor',
           'OpenFileProcessor']

TextMtd = IntEnum('TextMtd', 'DF TOK IDS')
text_extensions = ['.txt']

class LanguageModelLoader():
    "Create a dataloader with bptt slightly changing."
    def __init__(self, dataset:LabelList, bs:int=64, bptt:int=70, backwards:bool=False, shuffle:bool=False,
                 max_len:int=25):
        self.dataset,self.bs,self.bptt,self.backwards,self.shuffle = dataset,bs,bptt,backwards,shuffle
        self.first,self.i,self.iter = True,0,0
        self.n = len(np.concatenate(dataset.x.items)) // self.bs
        self.max_len,self.num_workers = max_len,0

    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None:
            yield LongTensor(getattr(self.dataset, 'item')).unsqueeze(1),LongTensor([0])
        idx = np.random.permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        self.data = self.batchify(np.concatenate([self.dataset.x.items[i] for i in idx]))
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.first and self.i == 0: self.first,seq_len = False,self.bptt + self.max_len
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                seq_len = min(seq_len, self.bptt + self.max_len)
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self) -> int: return (self.n-1) // self.bptt
    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

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
    samples = to_data(samples)
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[-len(s[0]):,i] = LongTensor(s[0])
        else:         res[:len(s[0]):,i] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])

def _get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=True, **kwargs):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, mark_fields=mark_fields),
            NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

class TextDataBunch(DataBunch):
    """General class to get a `DataBunch` for NLP. You should use one of its subclass, `TextLMDataBunch` or
    `TextClasDataBunch`."""
    
    def save(self, cache_name:PathOrStr='tmp'):
        "Save the `DataBunch` in `self.path/cache_name` folder."
        os.makedirs(self.path/cache_name, exist_ok=True)
        cache_path = self.path/cache_name
        pickle.dump(self.train_ds.vocab.itos, open(cache_path/f'itos.pkl', 'wb'))
        np.save(cache_path/f'train_ids.npy', self.train_ds.x.items)
        np.save(cache_path/f'train_lbl.npy', self.train_ds.y.items)
        np.save(cache_path/f'valid_ids.npy', self.valid_ds.x.items)
        np.save(cache_path/f'valid_lbl.npy', self.valid_ds.y.items)
        if self.test_dl is not None: np.save(cache_path/f'test_ids.npy', self.test_ds.x.items)
        save_texts(cache_path/'classes.txt', self.train_ds.classes)

    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None,
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None,
                 processor:PreProcessor=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a dictionary."
        src = ItemLists(path, TextList(train_ids, vocab, path=path, processor=[]),
                        TextList(valid_ids, vocab, path=path, processor=[]))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_lists(train_lbls, valid_lbls, classes=classes, processor=[])
        if test_ids is not None: src.add_test(TextList(test_ids, vocab, path=path))
        src.valid.x.processor = ifnone(processor, [TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        return src.databunch(**kwargs)

    @classmethod
    def load(cls, path:PathOrStr, cache_name:PathOrStr='tmp', processor:PreProcessor=None, **kwargs):
        "Load a `TextDataBunch` from `path/cache_name`. `kwargs` are passed to the dataloader creation."
        cache_path = Path(path)/cache_name
        vocab = Vocab(pickle.load(open(cache_path/f'itos.pkl', 'rb')))
        train_ids,train_lbls = np.load(cache_path/f'train_ids.npy'), np.load(cache_path/f'train_lbl.npy')
        valid_ids,valid_lbls = np.load(cache_path/f'valid_ids.npy'), np.load(cache_path/f'valid_lbl.npy')
        test_ids = np.load(cache_path/f'test_ids.npy') if os.path.isfile(cache_path/f'test_ids.npy') else None
        classes = loadtxt_str(cache_path/'classes.txt')
        return cls.from_ids(path, vocab, train_ids, valid_ids, test_ids, train_lbls, valid_lbls, classes, processor, **kwargs)

    @classmethod#TODO: test
    def from_tokens(cls, path:PathOrStr, trn_tok:Collection[Collection[str]], trn_lbls:Collection[Union[int,float]],
                 val_tok:Collection[Collection[str]], val_lbls:Collection[Union[int,float]], vocab:Vocab=None,
                 tst_tok:Collection[Collection[str]]=None, classes:Collection[Any]=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from tokens and labels."
        processor = _get_processor(tokenizer=None, vocab=vocab, **kwargs)[1]
        src = ItemLists(path, TextList(trn_tok, path=path, processor=processor),
                        TextList(valid_tok, path=path, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_lists(trn_lbls, val_lbls)
        if test_tok is not None: src.add_test(TextList(tst_tok, path=path))
        return src.databunch(**kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1, 
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, **kwargs)
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes, sep=label_delim)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name, valid_pct:float=0.2, test:Optional[str]=None,
                 tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, header = 'infer', text_cols:IntsOrStrs=1, 
                 label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from texts in csv files."
        df = pd.read_csv(Path(path)/csv_name, header=header)
        idx = np.random.permutation(len(df))
        cut = int(valid_pct * len(df)) + 1
        train_df, valid_df = df[cut:], df[:cut]
        test_df = None if test is None else pd.read_csv(Path(path)/test, header=header)
        return cls.from_df(path, train_df, valid_df, test_df, tokenizer, vocab, classes, text_cols, 
                           label_cols, label_delim, **kwargs)

    @classmethod#TODO: test
    def from_folder(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                    classes:Collection[Any]=None, tokenizer:Tokenizer=None, vocab:Vocab=None, **kwargs):
        "Create a `TextDataBunch` from text files in folders."
        path = Path(path)
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, **kwargs)
        src = (TextFilesList.from_folder(path)
                            .split_by_folder(train=train, valid=valid)
                            .label_from_folder(classes=classes))
        if test is not None: src.add_test_folder(path/test)
        return src.databunch(**kwargs)

def _treat_html(o:str)->str:
    return o.replace('\n','\\n')

#TODO: refactor common bit with tabular method of the same name
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

    #TODO: see if we can get rid of that later
    def show_batch(self, sep=' ', ds_type:DatasetType=DatasetType.Train, rows:int=10, max_len:int=100):
        "Show `rows` texts from a batch of `ds_type`, tokens are joined with `sep`, truncated at `max_len`."
        from IPython.display import display, HTML
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
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs//2, sampler=train_sampler, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            sampler = SortSampler(ds.x, key=lambda t: len(ds[t][0].data))
            dataloaders.append(DataLoader(ds, batch_size=bs, sampler=sampler, **kwargs))
        return cls(*dataloaders, path=path, collate_fn=collate_fn)

def open_text(fn:PathOrStr, enc='utf-8'):
    "Read the text in `fn`."
    with open(fn,'r', encoding = enc) as f: return ''.join(f.readlines())

class Text(ItemBase):
    def __init__(self, ids, text): self.data,self.text = ids,text
    def __str__(self):  return str(self.text)

    def show_batch(self, idxs:Collection[int], rows:int, ds:Dataset, max_len:int=50)->None:
        "Show the texts in `idx` on a few `rows` from `ds`. `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        items = [['text', 'label']]
        for i in idxs[:rows]:
            x,y = ds[i]
            txt_x = ' '.join(x.text.split(' ')[:max_len])
            items.append([str(txt_x), str(y)])
        display(HTML(_text2html_table(items, [90,10])))

class LMLabel(CategoryList):
    def predict(self, res): return res
    
class TokenizeProcessor(PreProcessor):
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, mark_fields:bool=True):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields

    def process_one(self, item):  return self.tokenizer._process_all_1([item])[0]
    def process(self, ds):
        ds.items = _join_texts(ds.items, self.mark_fields)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

class NumericalizeProcessor(PreProcessor):
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=2):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)

class OpenFileProcessor(PreProcessor):
    def process_one(self,item):
        return open_text(item) if isinstance(item, Path) else item
    
class TextList(ItemList):
    _bunch = TextClasDataBunch
    _processor = [TokenizeProcessor, NumericalizeProcessor]

    def __init__(self, items:Iterator, vocab:Vocab=None, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab = vocab

    def new(self, items:Iterator, **kwargs)->'NumericalizedTextList':
        return super().new(items=items, vocab=self.vocab, **kwargs)

    def get(self, i):
        o = super().get(i)
        return Text(o, self.vocab.textify(o))

    def label_for_lm(self, **kwargs):
        "A special labelling method for language models."
        self.__class__ = LMTextList
        return self.label_const(0, label_cls=LMLabel)
    
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=text_extensions, vocab:Vocab=None, 
                    processor:PreProcessor=None, **kwargs)->'TextList':
        "Get the list of files in `path` that have a text suffix. `recurse` determines if we search subfolders."
        processor = ifnone(processor, [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        return super().from_folder(path=path, extensions=extensions, processor=processor, **kwargs)

class LMTextList(TextList):
    _bunch = TextLMDataBunch
    
def _join_texts(texts:Collection[str], mark_fields:bool=True):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{FLD} {1} ' + df[0] if mark_fields else df[txt_cols[0]]
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i]
    return text_col.values