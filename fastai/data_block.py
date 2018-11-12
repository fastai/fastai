from .torch_core import *
from .basic_data import *

__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'LabelList', 'ItemLists', 'get_files', 'create_sdata', 'LabelLists']

def _decode(df):
    return np.array([[df.columns[i] for i,t in enumerate(x) if t==1] for x in df.values], dtype=np.object)

def _maybe_squeeze(arr):
    "Squeeze array dimensions but avoid squeezing a 1d-array containing a string."
    return (arr if is1d(arr) else np.squeeze(arr))

def _extract_input_labels(df:pd.DataFrame, input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1, is_fnames:bool=False,
                      label_delim:str=None, suffix:Optional[str]=None):
    """Get image file names in `fn_col` by adding `suffix` and labels in `label_col` from `df`.
    If `label_delim` is specified, splits the values in `label_col` accordingly.  """
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

class ItemList():
    _bunch = DataBunch

    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    def __init__(self, items:Iterator, create_func:Callable=None, path:PathOrStr='.',
                 label_cls:Callable=None, xtra:Any=None):
        self.items,self.create_func,self.path = np.array(list(items)),create_func,Path(path)
        self._label_cls,self.xtra = label_cls,xtra
        self._label_list,self._split = LabelList,ItemLists
        self.__post_init__()

    def __post_init__(self): pass
    def __len__(self)->int: return len(self.items)
    def __repr__(self)->str: return f'{self.__class__.__name__} ({len(self)} items)\n{self.items}\nPath: {self.path}'
    def get(self, i)->Any:
        item = self.items[i]
        return self.create_func(item) if self.create_func else item

    def new(self, items:Iterator, create_func:Callable=None, **kwargs)->'ItemList':
        create_func = ifnone(create_func, self.create_func)
        return self.__class__(items=items, create_func=create_func, path=self.path, **kwargs)

    def __getitem__(self,idxs:int)->Any:
        if isinstance(try_int(idxs), int): return self.get(idxs)
        else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))

    def preprocess(self, **kwargs): pass

    @classmethod
    def from_folder(cls, path:PathOrStr, create_func:Callable=None, extensions:Collection[str]=None, recurse=True)->'ItemList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions, recurse=recurse), create_func=create_func, path=path)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', create_func:Callable=None, col:IntsOrStrs=0)->'ItemList':
        "Get the list of inputs in the `col` of `path/csv_name`."
        inputs = df.iloc[:,df_names_to_idx(col, df)]
        res = cls(create_func=create_func, items=_maybe_squeeze(inputs.values), path=path, xtra = df)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, create_func:Callable=None, col:IntsOrStrs=0, header:str='infer')->'ItemList':
        "Get the list of inputs in the `col`of `path/csv_name`."
        df = pd.read_csv(path/csv_name, header=header)
        return cls.from_df(df, path=path, create_func=create_func, col=col)

    #Not adapted
    @classmethod
    def from_csvs(cls, path:PathOrStr, csv_fnames:Collection[PathOrStr], input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1,
                  header:str='infer')->'LabelList':
        """Create in `path` by reading `input_cols` and `label_cols` in the csvs in `path/csv_names`
        opened with `header`. If `label_delim` is specified, splits the tags in `label_cols` accordingly.  """
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
        train_idx = [i for i in range_of(self) if i not in valid_idx]
        return self.split_by_idxs(train_idx, valid_idx)

    def _get_by_folder(self, name):
        return [i for i in range_of(self)
                if self.items[i].relative_to(self.path).parts[0] == name]

    def split_by_folder(self, train:str='train', valid:str='valid')->'ItemLists':
        "Split the data depending on the folder (`train` or `valid`) in which the filenames are."
        return self.split_by_idxs(self._get_by_folder(train), self._get_by_folder(valid))

    def random_split_by_pct(self, valid_pct:float=0.2, seed:int=None)->'ItemLists':
        "Split the items randomly by putting `valid_pct` in the validation set."
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

    def label_cls(self, labels, lc=None):
        if lc is not None:              return lc
        if self._label_cls is not None: return self._label_cls
        it = try_int(index_row(labels,0))
        if isinstance(it, (str,int)):   return CategoryList
        if isinstance(it, Collection):  return MultiCategoryList
        return self.__class__

    def label_from_list(self, labels:Iterator, label_cls:Callable=None, template:Callable=None, **kwargs)->'LabelList':
        label_cls = self.label_cls(labels, label_cls)
        y_bld = label_cls if template is None else template.new
        return self._label_list(x=self, y=y_bld(labels, **kwargs))

    def label_from_df(self, cols:IntsOrStrs=1, sep=None, **kwargs):
        labels = _maybe_squeeze(self.xtra.iloc[:,df_names_to_idx(cols, self.xtra)])
        return self.label_from_list(labels, sep=sep, **kwargs)

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
        return self.__class__(items, ifnone(classes, self.classes), **kwargs)

    def get(self, i):
        o = super().get(i)
        return self._item_cls.create(o, self.class2idx)

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
            return self
        return _inner

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_fname:PathOrStr, input_cols:IntsOrStrs=0, label_cols:IntsOrStrs=1,
                 valid_col:int=2, header:str='infer')->'ItemLists':
        """Create a `ItemLists` in `path` from the csv in `path/csv_name` read with `header`. Take the inputs from
        `input_cols`, the labels from `label_cols` and split by `valid_col` (`True` indicates valid set)."""
        df = pd.read_csv(path/csv_fname, header=header)
        val_idx = df.iloc[:,valid_col].nonzero()[0]
        return LabelList.from_df(path, df, input_cols, label_cols).split_by_idx(val_idx)

    @property
    def lists(self):
        res = [self.train,self.valid]
        if self.test is not None: res.append(self.test)
        return res

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

    def preprocess(self, **kwargs):
        self.train.preprocess(**kwargs)
        kwargs = {**kwargs, **getattr(self.train, 'preprocess_kwargs', {})}
        for ds in self.lists[1:]: ds.preprocess(**kwargs)
        self.valid.pp_kwargs = kwargs
        return self

class LabelLists(ItemLists):
    def databunch(self, path:PathOrStr=None, **kwargs)->'ImageDataBunch':
        "Create an `ImageDataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `ImageDataBunch.create`."
        path = Path(ifnone(path, self.path))
        return self.x._bunch.create(self.train, self.valid, test_ds=self.test, path=path, **kwargs)

    def add_test(self, items:Iterator, label:Any=None):
        "Add test set containing items from `items` and an arbitrary `label`"
        # if no label passed, use label of first training item
        if label is None: label=self.train[0][1]
        v = self.valid
        x = v.x.new(items)
        y = v.y.new([label for _ in range_of(x)])
        self.test = self.valid.new(x, y)
        return self

    def add_test_folder(self, test_folder:str='test', label:Any=None):
        "Add test set containing items from folder `test_folder` and an arbitrary `label`."
        items = self.x.__class__.from_folder(self.path/test_folder)
        return self.add_test(items.items, label=label)

class LabelList(Dataset):
    "A list of inputs and labels. Contain methods to split it in `ItemLists`."
    def __init__(self, x:ItemList, y:ItemList, tfms:TfmList=None, tfm_y:bool=False, pp_kwargs=None, **kwargs):
        self.x,self.y,self.tfm_y,self.pp_kwargs = x,y,tfm_y,pp_kwargs
        self.y.x = x
        if pp_kwargs is not None: self.x.preprocess(**pp_kwargs)
        self.transform(tfms, **kwargs)

    def __len__(self)->int: return len(self.x)
    def __repr__(self)->str: return f'{self.__class__.__name__}\ny: {self.y}\nx: {self.x}'

    @property
    def c(self): return self.y.c

    def new(self, x, y)->'LabelList':
        return self.__class__(x, y, tfms=self.tfms, tfm_y=self.tfm_y, pp_kwargs=self.pp_kwargs, **self.tfmargs)

    def __getattr__(self,k:str)->Any:
        res = getattr(self.x, k, None)
        return res if res is not None else getattr(self.y, k)

    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        if isinstance(try_int(idxs), int):
            x = self.x[idxs]
            y = self.y[idxs]
            x = x.apply_tfms(self.tfms, **self.tfmargs)
            if self.tfm_y: y = y.apply_tfms(self.tfms, **{**self.tfmargs, 'do_resolve':False})
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])

    @classmethod
    def from_lists(cls, path:PathOrStr, inputs, labels)->'LabelList':
        "Create a `LabelDataset` in `path` with `inputs` and `labels`."
        inputs,labels = np.array(inputs),np.array(labels)
        return cls(np.concatenate([inputs[:,None], labels[:,None]], 1), path)

    def transform(self, tfms:TfmList, tfm_y:bool=None, **kwargs):
        self.tfms,self.tfmargs = tfms,kwargs
        if tfm_y is not None: self.tfm_y=tfm_y
        return self

def create_sdata(sdata_cls, path:PathOrStr, train_x:Collection, train_y:Collection, valid_x:Collection,
                 valid_y:Collection, test_x:Collection=None):
    train = LabelList.from_lists(path, train_x, train_y)
    valid = LabelList.from_lists(path, valid_x, valid_y)
    test = ItemList(test_x, path).label_const(0) if test_x is not None else None
    return ItemLists(path, train, valid, test)

