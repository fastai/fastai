from .torch_core import *
from .basic_data import *


__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'LabelList', 'ItemLists', 'get_files',
           'PreProcessor', 'LabelLists']

def _decode(df):
    return np.array([[df.columns[i] for i,t in enumerate(x) if t==1] for x in df.values], dtype=np.object)

def _maybe_squeeze(arr):
    "Squeeze array dimensions but avoid squeezing a 1d-array containing a string."
    return (arr if is1d(arr) else np.squeeze(arr))

def _extract_input_labels(df:pd.DataFrame, input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1, is_fnames:bool=False,
                      label_delim:str=None, suffix:Optional[str]=None):
    "Get image file names in `fn_col` by adding `suffix` and labels in `label_col` from `df`."
    assert label_delim is None or not isinstance(label_cols, Iterable) or len(label_cols) == 1
    labels = df.iloc[:,df_names_to_idx(label_cols, df)]
    if label_delim: labels = np.array(list(csv.reader(labels.iloc[:,0], delimiter=label_delim)))
    else:
        if isinstance(label_cols, Iterable) and len(label_cols) > 1: labels = _decode(labels)
        else: labels = _maybe_squeeze(labels.values)
    inputs = df.iloc[:,df_names_to_idx(input_cols, df)]
    if is_fnames: inputs = inputs.iloc[:,0].str.lstrip()
    if suffix: inputs = inputs + suffix
    return _maybe_squeeze(inputs.values), labels

def get_files(c:PathOrStr, extensions:Collection[str]=None, recurse:bool=False)->FilePathList:
    "Return list of files in `c` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
    return [o for o in Path(c).glob('**/*' if recurse else '*')
            if not o.name.startswith('.') and not o.is_dir()
            and (extensions is None or (o.suffix.lower() in extensions))]

class PreProcessor():
    def process_one(self, item):      return item
    def process(self, ds:Collection): return self

class ItemList():
    _bunch = DataBunch
    _processor = PreProcessor

    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    def __init__(self, items:Iterator, create_func:Callable=None, path:PathOrStr='.',
                 label_cls:Callable=None, xtra:Any=None, processor:PreProcessor=None):
        self.items,self.create_func,self.path = np.array(list(items), dtype=object),create_func,Path(path)
        self._label_cls,self.xtra,self.processor = label_cls,xtra,processor
        self._label_list,self._split = LabelList,ItemLists
        self.__post_init__()

    def __post_init__(self): pass
    def __len__(self)->int: return len(self.items) or 1
    def __repr__(self)->str: return f'{self.__class__.__name__} ({len(self)} items)\n{self.items}\nPath: {self.path}'
    def get(self, i)->Any:
        item = self.items[i]
        return self.create_func(item) if self.create_func else item

    def process(self, processor=None):
        if processor is not None: self.processor = processor
        if not is_listy(self.processor): self.processor = [self.processor]
        for p in self.processor: p.process(self)
        return self

    def process_one(self, item, processor=None):
        if processor is not None: self.processor = processor
        if not is_listy(self.processor): self.processor = [self.processor]
        for p in self.processor: item = p.process_one(item)
        return item

    def predict(self, res):
        "Called at the end of `Learn.predict`; override for optional post-processing"
        return res

    def new(self, items:Iterator, create_func:Callable=None, **kwargs)->'ItemList':
        create_func = ifnone(create_func, self.create_func)
        return self.__class__(items=items, create_func=create_func, path=self.path, processor=self.processor, **kwargs)

    def __getitem__(self,idxs:int)->Any:
        if isinstance(try_int(idxs), int): return self.get(idxs)
        else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))

    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=None, recurse=True, **kwargs)->'ItemList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions, recurse=recurse), path=path, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', col:IntsOrStrs=0, **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `col` of `df`."
        inputs = df.iloc[:,df_names_to_idx(col, df)]
        res = cls(items=_maybe_squeeze(inputs.values), path=path, xtra = df, **kwargs)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, col:IntsOrStrs=0, header:str='infer', **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `col` of `path/csv_name` opened with `header`."
        df = pd.read_csv(path/csv_name, header=header)
        return cls.from_df(df, path=path, col=col, **kwargs)

    #Not adapted
    @classmethod
    def from_csvs(cls, path:PathOrStr, csv_fnames:Collection[PathOrStr], input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1,
                  header:str='infer', **kwargs)->'LabelList':
        "Create in `path` by reading `input_cols` and `label_cols` in csvs in `path/csv_names` opened with `header`."
        return cls(np.concatenate([cls.from_csv(path, fname, input_cols, label_cols).items for fname in csv_fnames]), path)

    def filter_by_func(self, func:Callable)->'ItemList':
        self.items = array([o for o in self.items if func(o)])
        return self

    def filter_by_folder(self, include=None, exclude=None):
        include,exclude = listify(include),listify(exclude)
        def _inner(o):
            n = o.relative_to(self.path).parts[0]
            if include and not n in include: return False
            if exclude and     n in exclude: return False
            return True
        return self.filter_by_func(_inner)

    def split_by_list(self, train, valid):
        return self._split(self.path, train, valid)

    def split_by_idxs(self, train_idx, valid_idx):
        return self.split_by_list(self[train_idx], self[valid_idx])

    def split_by_idx(self, valid_idx:Collection[int])->'ItemLists':
        "Split the data according to the indexes in `valid_idx`."
        train_idx = [i for i in range_of(self.items) if i not in valid_idx]
        return self.split_by_idxs(train_idx, valid_idx)

    def _get_by_folder(self, name):
        return [i for i in range_of(self)
                if self.items[i].relative_to(self.path).parts[0] == name]

    def split_by_folder(self, train:str='train', valid:str='valid')->'ItemLists':
        "Split the data depending on the folder (`train` or `valid`) in which the filenames are."
        return self.split_by_idxs(self._get_by_folder(train), self._get_by_folder(valid))

    def random_split_by_pct(self, valid_pct:float=0.2, seed:int=None)->'ItemLists':
        "Split the items randomly by putting `valid_pct` in the validation set. Set the `seed` in numpy if passed."
        if seed is not None: np.random.seed(seed)
        rand_idx = np.random.permutation(range_of(self))
        cut = int(valid_pct * len(self))
        return self.split_by_idx(rand_idx[:cut])

    def split_by_valid_func(self, func:Callable)->'ItemLists':
        "Split the data by result of `func` (which returns `True` for validation set)"
        valid_idx = [i for i,o in enumerate(self.items) if func(o)]
        return self.split_by_idx(valid_idx)

    def split_by_files(self, valid_names:'ItemList')->'ItemLists':
        "Split the data by using the names in `valid_names` for validation."
        #valid_idx = [i for i,o in enumerate(self.items) if o[0] in valid_names]
        return self.split_by_valid_func(lambda o: o.name in valid_names)

    def split_by_fname_file(self, fname:PathOrStr, path:PathOrStr=None)->'ItemLists':
        "Split the data by using the file names in `fname` for the validation set. `path` will override `self.path`."
        path = Path(ifnone(path, self.path))
        valid_names = loadtxt_str(self.path/fname)
        return self.split_by_files(valid_names)

    def split_from_df(self, col:IntsOrStrs=2):
        "Split the data from the `col` in the dataframe in `self.xtra`."
        valid_idx = np.where(self.xtra.iloc[:,df_names_to_idx(col, self.xtra)])[0]
        return self.split_by_idx(valid_idx)

    def label_cls(self, labels, lc=None):
        if lc is not None:              return lc
        if self._label_cls is not None: return self._label_cls
        it = try_int(index_row(labels,0))
        if isinstance(it, (str,int)):   return CategoryList
        if isinstance(it, Collection):  return MultiCategoryList
        return self.__class__

    def label_from_list(self, labels:Iterator, label_cls:Callable=None, template:Callable=None, **kwargs)->'LabelList':
        "Label `self.items` with `labels` using `label_cls` and optionally `template`."
        labels = array(labels, dtype=object)
        label_cls = self.label_cls(labels, label_cls)
        y_bld = label_cls if template is None else template.new
        y = y_bld(labels, **kwargs)
        if self.__class__.__name__.startswith('Text'):
            filt = array([o is None for o in y])
            if filt.sum()>0: self,y = self[~filt],y[~filt]
        return self._label_list(x=self, y=y)

    def label_from_df(self, cols:IntsOrStrs=1, sep=None, **kwargs):
        "Label `self.items` from the values in `cols` in `self.xtra`. If `sep` is passed, will split the labels accordingly."
        labels = _maybe_squeeze(self.xtra.iloc[:,df_names_to_idx(cols, self.xtra)])
        label_cls = None if sep is None else MultiCategoryList
        return self.label_from_list(labels, label_cls=label_cls, sep=sep, **kwargs)

    def label_const(self, const:Any=0, **kwargs)->'LabelList':
        "Label every item with `const`."
        return self.label_from_func(func=lambda o: const, **kwargs)

    def label_from_func(self, func:Callable, **kwargs)->'LabelList':
        "Apply `func` to every input to get its label."
        return self.label_from_list([func(o) for o in self.items], **kwargs)

    def label_from_folder(self, **kwargs)->'LabelList':
        "Give a label to each filename depending on its folder."
        return self.label_from_func(func=lambda o: o.parent.name, **kwargs)

    def label_from_re(self, pat:str, full_path:bool=False, **kwargs)->'LabelList':
        "Apply the re in `pat` to determine the label of every filename.  If `full_path`, search in the full name."
        pat = re.compile(pat)
        def _inner(o):
            s = str(o if full_path else o.name)
            res = pat.search(s)
            assert res,f'Failed to find "{pat}" in "{s}"'
            return res.group(1)
        return self.label_from_func(_inner, **kwargs)

class CategoryList(ItemList):
    _item_cls=Category
    def __init__(self, items:Iterator, classes:Collection=None, sep=None, **kwargs):
        super().__init__(items, **kwargs)
        if classes is None: classes = uniqueify(items)
        self.classes = classes
        self.class2idx = {v:k for k,v in enumerate(self.classes)}
        self.c = len(classes)
        self.loss_func = F.cross_entropy

    def new(self, items, classes=None, **kwargs):
        return super().new(items, classes=ifnone(classes, self.classes), **kwargs)

    def get(self, i):
        o = super().get(i)
        return self._item_cls.create(o, self.class2idx)

    def predict(self, res):
        pred_max = res[0].argmax()
        return self.classes[pred_max],pred_max,res[0]

class MultiCategoryList(CategoryList):
    _item_cls=MultiCategory
    def __init__(self, items:Iterator, classes:Collection=None, sep=None, **kwargs):
        if sep is not None: items = array(list(csv.reader(items, delimiter=sep)))
        if classes is None: classes = uniqueify(np.concatenate(items))
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = F.binary_cross_entropy_with_logits

class ItemLists():
    "A `ItemList` for each of `train` and `valid` (optional `test`)"
    def __init__(self, path:PathOrStr, train:ItemList, valid:ItemList, test:ItemList=None):
        self.path,self.train,self.valid,self.test = Path(path),train,valid,test
        if isinstance(self.train, LabelList): self.__class__ = LabelLists

    def __repr__(self)->str:
        return f'{self.__class__.__name__};\nTrain: {self.train};\nValid: {self.valid};\nTest: {self.test}'

    def __getattr__(self, k):
        ft = getattr(self.train, k)
        if not isinstance(ft, Callable): return ft
        fv = getattr(self.valid, k)
        assert isinstance(fv, Callable)
        def _inner(*args, **kwargs):
            self.train = ft(*args, **kwargs)
            assert isinstance(self.train, LabelList)
            self.valid = fv(*args, template=self.train.y, **kwargs)
            self.__class__ = LabelLists
            self.process()
            return self
        return _inner

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_fname:PathOrStr, input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1,
                 valid_col:int=2, header:str='infer')->'ItemLists':
        "Create in `path` from csv in `path/csv_name` with `header`. Inputs from `input_cols`, labels `label_cols` split by `valid_col`"
        df = pd.read_csv(path/csv_fname, header=header)
        val_idx = df.iloc[:,valid_col].nonzero()[0]
        return LabelList.from_df(path, df, input_cols, label_cols).split_by_idx(val_idx)

    @property
    def lists(self):
        res = [self.train,self.valid]
        if self.test is not None: res.append(self.test)
        return res

    def label_from_lists(self, train_labels:Iterator, valid_labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        label_cls = self.train.label_cls(train_labels, label_cls)
        self.train = self.train._label_list(x=self.train, y=label_cls(train_labels, **kwargs))
        self.valid = self.valid._label_list(x=self.valid, y=self.train.y.new(valid_labels, **kwargs))
        self.__class__ = LabelLists
        self.process()
        return self

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        if not tfms: return self
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

class LabelLists(ItemLists):
    def get_processors(self):
        xp = ifnone(self.train.x.processor, self.train.x._processor())
        yp = ifnone(self.train.y.processor, self.train.y._processor())
        return xp,yp

    def process(self):
        xp,yp = self.get_processors()
        for ds in self.lists: ds.process(xp, yp)
        return self

    def databunch(self, path:PathOrStr=None, **kwargs)->'ImageDataBunch':
        "Create an `DataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        path = Path(ifnone(path, self.path))
        return self.x._bunch.create(self.train, self.valid, test_ds=self.test, path=path, **kwargs)

    def add_test(self, items:Iterator, label:Any=None):
        "Add test set containing items from `items` and an arbitrary `label`"
        # if no label passed, use label of first training item
        if label is None: label = str(self.train[0][1])
        labels = [label for _ in range_of(items)]
        if isinstance(items, ItemList): self.test = self.valid.new(items.items, labels, xtra=items.xtra)
        else: self.test = self.valid.new(items, labels)
        return self

    def add_test_folder(self, test_folder:str='test', label:Any=None):
        "Add test set containing items from folder `test_folder` and an arbitrary `label`."
        items = self.x.__class__.from_folder(self.path/test_folder)
        return self.add_test(items.items, label=label)

class LabelList(Dataset):
    "A list of inputs and labels. Contain methods to split it in `ItemLists`."
    def __init__(self, x:ItemList, y:ItemList, tfms:TfmList=None, tfm_y:bool=False, **kwargs):
        self.x,self.y,self.tfm_y = x,y,tfm_y
        self.y.x = x
        self.item=None
        self.transform(tfms, **kwargs)

    def __len__(self)->int: return len(self.x) if self.item is None else 1
    def set_item(self,item): self.item = self.x.process_one(item)
    def clear_item(self): self.item = None
    def __repr__(self)->str: return f'{self.__class__.__name__}\ny: {self.y}\nx: {self.x}'
    def predict(self, res): return self.y.predict(res)

    @property
    def c(self): return self.y.c

    def new(self, x, y, **kwargs)->'LabelList':
        if isinstance(x, ItemList):
            return self.__class__(x, y, tfms=self.tfms, tfm_y=self.tfm_y, **self.tfmargs)
        else:
            return self.new(self.x.new(x, **kwargs), self.y.new(y, **kwargs)).process()

    def __getattr__(self,k:str)->Any:
        res = getattr(self.x, k, None)
        return res if res is not None else getattr(self.y, k)

    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        if isinstance(try_int(idxs), int):
            if self.item is None: x,y = self.x[idxs],self.y[idxs]
            else:                 x,y = self.item   ,0
            if self.tfms:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
                if self.tfm_y and self.item is None:
                    y = y.apply_tfms(self.tfms, **{**self.tfmargs, 'do_resolve':False})
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])

    def process(self, xp=None, yp=None):
        "Launch the preprocessing on `xp` and `yp`."
        self.x.process(xp)
        self.y.process(yp)
        return self

    @classmethod
    def from_lists(cls, path:PathOrStr, inputs, labels)->'LabelList':
        "Create a `LabelDataset` in `path` with `inputs` and `labels`."
        inputs,labels = np.array(inputs),np.array(labels)
        return cls(np.concatenate([inputs[:,None], labels[:,None]], 1), path)

    def transform(self, tfms:TfmList, tfm_y:bool=None, **kwargs):
        "Set the `tfms` and `` tfm_y` value to be applied to the inputs and targets."
        self.tfms,self.tfmargs = tfms,kwargs
        if tfm_y is not None: self.tfm_y=tfm_y
        return self
