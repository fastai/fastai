from .torch_core import *
from .basic_data import *
from .layers import MSELossFlat

__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'MultiCategoryProcessor', 'LabelList', 'ItemLists', 'get_files',
           'PreProcessor', 'LabelLists', 'FloatList', 'CategoryProcessor']

def _decode(df):
    return np.array([[df.columns[i] for i,t in enumerate(x) if t==1] for x in df.values], dtype=np.object)

def _maybe_squeeze(arr): return (arr if is1d(arr) else np.squeeze(arr))

def get_files(c:PathOrStr, extensions:Collection[str]=None, recurse:bool=False)->FilePathList:
    "Return list of files in `c` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
    return [o for o in Path(c).glob('**/*' if recurse else '*')
            if not o.name.startswith('.') and not o.is_dir()
            and (extensions is None or (o.suffix.lower() in extensions))]

class PreProcessor():
    def __init__(self, ds:Collection=None):  self.ref_ds = ds
    def process_one(self, item:Any):         return item
    def process(self, ds:Collection):        ds.items = array([self.process_one(item) for item in ds.items])

class ItemList():
    _bunch = DataBunch
    _processor = PreProcessor

    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    def __init__(self, items:Iterator, create_func:Callable=None, path:PathOrStr='.',
                 label_cls:Callable=None, xtra:Any=None, processor:PreProcessor=None, **kwargs):
        self.items,self.create_func,self.path = array(items, dtype=object),create_func,Path(path)
        self._label_cls,self.xtra,self.processor = label_cls,xtra,processor
        self._label_list,self._split = LabelList,ItemLists
        self.__post_init__()

    def __post_init__(self): pass
    def __len__(self)->int: return len(self.items) or 1
    def __repr__(self)->str:
        items = [self[i] for i in range(min(5,len(self)))]
        return f'{self.__class__.__name__} ({len(self)} items)\n{items}...\nPath: {self.path}'
    def get(self, i)->Any:
        item = self.items[i]
        return self.create_func(item) if self.create_func else item

    def process(self, processor=None):
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: p.process(self)
        return self

    def process_one(self, item, processor=None):
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: item = p.process_one(item)
        return item

    def predict(self, res):
        "Called at the end of `Learn.predict`; override for optional post-processing"
        return res

    def new(self, items:Iterator, create_func:Callable=None, processor:PreProcessor=None, **kwargs)->'ItemList':
        create_func = ifnone(create_func, self.create_func)
        processor = ifnone(processor, self.processor)
        return self.__class__(items=items, create_func=create_func, path=self.path, processor=processor, **kwargs)

    def __getitem__(self,idxs:int)->Any:
        if isinstance(try_int(idxs), int): return self.get(idxs)
        else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))

    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=None, recurse=True, **kwargs)->'ItemList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions, recurse=recurse), path=path, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', cols:IntsOrStrs=0, **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        inputs = df.iloc[:,df_names_to_idx(cols, df)]
        res = cls(items=_maybe_squeeze(inputs.values), path=path, xtra = df, **kwargs)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, cols:IntsOrStrs=0, header:str='infer', **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `path/csv_name` opened with `header`."
        df = pd.read_csv(path/csv_name, header=header)
        return cls.from_df(df, path=path, cols=cols, **kwargs)

    def filter_by_func(self, func:Callable)->'ItemList':
        "Only keeps elements for which `func` returns `True`."
        self.items = array([o for o in self.items if func(o)])
        return self

    def filter_by_folder(self, include=None, exclude=None):
        "Only keep filenames in `include` folder or reject the ones in `exclude`."
        include,exclude = listify(include),listify(exclude)
        def _inner(o):
            n = o.relative_to(self.path).parts[0]
            if include and not n in include: return False
            if exclude and     n in exclude: return False
            return True
        return self.filter_by_func(_inner)

    def split_by_list(self, train, valid):
        "Split the data between `train` and `valid`."
        return self._split(self.path, train, valid)

    def split_by_idxs(self, train_idx, valid_idx):
        "Split the data between `train_idx` and `valid_idx`."
        return self.split_by_list(self[train_idx], self[valid_idx])

    def split_by_idx(self, valid_idx:Collection[int])->'ItemLists':
        "Split the data according to the indexes in `valid_idx`."
        #train_idx = [i for i in range_of(self.items) if i not in valid_idx]
        train_idx = np.setdiff1d(arange_of(self.items), valid_idx)
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

    def label_cls(self, labels, label_cls:Callable=None, sep:str=None, **kwargs):
        if label_cls is not None:               return label_cls
        if self._label_cls is not None:         return self._label_cls
        it = index_row(labels,0)
        if sep is not None:                     return MultiCategoryList
        if isinstance(it, (float, np.float32)): return FloatList
        if isinstance(try_int(it), (str,int)):  return CategoryList
        if isinstance(it, Collection):          return MultiCategoryList
        return self.__class__

    def label_from_list(self, labels:Iterator, **kwargs)->'LabelList':
        "Label `self.items` with `labels` using `label_cls`"
        labels = array(labels, dtype=object)
        label_cls = self.label_cls(labels, **kwargs)
        y = label_cls(labels, **kwargs)
        res = self._label_list(x=self, y=y)
        return res

    def label_from_df(self, cols:IntsOrStrs=1, **kwargs):
        "Label `self.items` from the values in `cols` in `self.xtra`."
        labels = _maybe_squeeze(self.xtra.iloc[:,df_names_to_idx(cols, self.xtra)])
        return self.label_from_list(labels, **kwargs)

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

class CategoryProcessor(PreProcessor):
    def __init__(self, ds:ItemList): self.create_classes(ds.classes)

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {v:k for k,v in enumerate(classes)}

    def generate_classes(self, items): return uniqueify(items)
    def process_one(self,item): return self.c2i.get(item,None)

    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i
        super().process(ds)

class CategoryListBase(ItemList):
    def __init__(self, items:Iterator, classes:Collection=None,**kwargs):
        self.classes=classes
        super().__init__(items, **kwargs)
        
    @property
    def c(self): return len(self.classes)

    def new(self, items, classes=None, **kwargs):
        return super().new(items, classes=ifnone(classes, self.classes), **kwargs)

class CategoryList(CategoryListBase):
    _item_cls=Category
    _processor=CategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = F.cross_entropy

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return self._item_cls(o, self.classes[o])

    def predict(self, res):
        pred_max = res[0].argmax()
        return self.classes[pred_max],pred_max,res[0]

class MultiCategoryProcessor(CategoryProcessor):
    def process_one(self,item): return [self.c2i.get(o,None) for o in item]

    def generate_classes(self, items):
        classes = set()
        for c in items: classes = classes.union(set(c))
        return list(classes)

class MultiCategoryList(CategoryListBase):
    _item_cls=MultiCategory
    _processor=MultiCategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, sep:str=None, **kwargs):
        if sep is not None: items = array(csv.reader(items, delimiter=sep))
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = F.binary_cross_entropy_with_logits

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return self._item_cls(one_hot(o, self.c), [self.classes[p] for p in o], o)

class FloatList(ItemList):
    _item_cls=FloatItem
    def __init__(self, items:Iterator, log:bool=False, **kwargs):
        super().__init__(np.array(items, dtype=np.float32), **kwargs)
        self.log = log
        self.c = self.items.shape[1] if len(self.items.shape) > 1 else 1
        self.loss_func = MSELossFlat()

    def new(self, items,**kwargs):
        return super().new(items, log=self.log, **kwargs)

    def get(self, i):
        o = super().get(i)
        return self._item_cls(log(o) if self.log else o)

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
            self.valid = fv(*args, **kwargs)
            self.__class__ = LabelLists
            self.process()
            return self
        return _inner

    @property
    def lists(self):
        res = [self.train,self.valid]
        if self.test is not None: res.append(self.test)
        return res

    def label_from_lists(self, train_labels:Iterator, valid_labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        "Use the labels in `train_labels` and `valid_labels` to label the data. `label_cls` will overwrite the default."
        label_cls = self.train.label_cls(train_labels, label_cls)
        self.train = self.train._label_list(x=self.train, y=label_cls(train_labels, **kwargs))
        self.valid = self.valid._label_list(x=self.valid, y=self.train.y.new(valid_labels, **kwargs))
        self.__class__ = LabelLists
        self.process()
        return self

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the train and validation set."
        if not tfms: return self
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

class LabelLists(ItemLists):
    def get_processors(self):
        procs_x,procs_y = listify(self.train.x._processor),listify(self.train.y._processor)
        xp = ifnone(self.train.x.processor, [p(ds=self.train.x) for p in procs_x])
        yp = ifnone(self.train.y.processor, [p(ds=self.train.y) for p in procs_y])
        return xp,yp

    def process(self):
        xp,yp = self.get_processors()
        for i,ds in enumerate(self.lists): ds.process(xp, yp, filter_missing_y=i==0)
        return self

    def databunch(self, path:PathOrStr=None, **kwargs)->'ImageDataBunch':
        "Create an `DataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        path = Path(ifnone(path, self.path))
        return self.x._bunch.create(self.train, self.valid, test_ds=self.test, path=path, **kwargs)

    def add_test(self, items:Iterator, label:Any=None):
        "Add test set containing items from `items` and an arbitrary `label`"
        # if no label passed, use label of first training item
        if label is None: label = self.train[0][1].obj
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

    def process(self, xp=None, yp=None, filter_missing_y:bool=False):
        "Launch the preprocessing on `xp` and `yp`."
        self.y.process(yp)
        if filter_missing_y and (getattr(self.x, 'filter_missing_y', None)):
            filt = array([o is None for o in self.y])
            if filt.sum()>0: self.x,self.y = self.x[~filt],self.y[~filt]
        self.x.process(xp)
        return self

    @classmethod
    def from_lists(cls, path:PathOrStr, inputs, labels)->'LabelList':
        "Create a `LabelList` in `path` with `inputs` and `labels`."
        inputs,labels = np.array(inputs),np.array(labels)
        return cls(np.concatenate([inputs[:,None], labels[:,None]], 1), path)

    def transform(self, tfms:TfmList, tfm_y:bool=None, **kwargs):
        "Set the `tfms` and `` tfm_y` value to be applied to the inputs and targets."
        self.tfms,self.tfmargs = tfms,kwargs
        if tfm_y is not None: self.tfm_y=tfm_y
        return self

