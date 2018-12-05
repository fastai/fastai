from .torch_core import *
from .basic_data import *
from .layers import *

__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'MultiCategoryProcessor', 'LabelList', 'ItemLists', 'get_files',
           'PreProcessor', 'LabelLists', 'FloatList', 'CategoryProcessor']

def _decode(df):
    return np.array([[df.columns[i] for i,t in enumerate(x) if t==1] for x in df.values], dtype=np.object)

def _maybe_squeeze(arr): return (arr if is1d(arr) else np.squeeze(arr))

def _get_files(parent, p, f, extensions):
    p = Path(p)#.relative_to(parent)
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or f'.{o.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path:PathOrStr, extensions:Collection[str]=None, recurse:bool=False,
              include:Optional[Collection[str]]=None)->FilePathList:
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for p,d,f in os.walk(path):
            # skip hidden dirs
            if include is not None: d[:] = [o for o in d if o in include]
            else:                   d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, path, f, extensions)

class PreProcessor():
    "Basic class for a processor that will be applied to items at the end of the data block API."
    def __init__(self, ds:Collection=None):  self.ref_ds = ds
    def process_one(self, item:Any):         return item
    def process(self, ds:Collection):        ds.items = array([self.process_one(item) for item in ds.items])

class ItemList():
    _bunch,_processor,_label_cls,_square_show = DataBunch,None,None,False

    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    def __init__(self, items:Iterator, path:PathOrStr='.',
                 label_cls:Callable=None, xtra:Any=None, processor:PreProcessor=None, x:'ItemList'=None, **kwargs):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x = array(items, dtype=object),x
        self.label_cls,self.xtra,self.processor = ifnone(label_cls,self._label_cls),xtra,processor
        self._label_list,self._split = LabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']
        self.__post_init__()

    def __post_init__(self): pass
    def __len__(self)->int: return len(self.items) or 1
    def get(self, i)->Any:
        "Subclass if you want to customize how to create item `i` from `self.items`."
        return self.items[i]
    def __repr__(self)->str:
        items = [self[i] for i in range(min(5,len(self.items)))]
        return f'{self.__class__.__name__} ({len(self)} items)\n{items}...\nPath: {self.path}'

    def process(self, processor=None):
        "Apply `processor` or `self.processor` to `self`."
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: p.process(self)
        return self

    def process_one(self, item, processor=None):
        "Apply `processor` or `self.processor` to `item`."
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: item = p.process_one(item)
        return item

    def analyze_pred(self, pred:Tensor):
        "Called on `pred` before `reconstruct` for additional preprocessing."
        return pred

    def reconstruct(self, t:Tensor, x:Tensor=None):
        "Reconstuct one of the underlying item for its data `t`."
        return self[0].reconstruct(t,x) if has_arg(self[0].reconstruct, 'x') else self[0].reconstruct(t)

    def new(self, items:Iterator, processor:PreProcessor=None, **kwargs)->'ItemList':
        "Create a new `ItemList` from `items`, keeping the same attributes."
        processor = ifnone(processor, self.processor)
        copy_d = {o:getattr(self,o) for o in self.copy_new}
        return self.__class__(items=items, processor=processor, **copy_d, **kwargs)

    def __getitem__(self,idxs:int)->Any:
        if isinstance(try_int(idxs), int): return self.get(idxs)
        else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))

    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=None, recurse=True,
                    include:Optional[Collection[str]]=None, **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        path = Path(path)
        return cls(get_files(path, extensions, recurse=recurse, include=include), path=path, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', cols:IntsOrStrs=0, **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        inputs = df.iloc[:,df_names_to_idx(cols, df)]
        res = cls(items=_maybe_squeeze(inputs.values), path=path, xtra = df, **kwargs)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, cols:IntsOrStrs=0, header:str='infer', **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `path/csv_name` opened with `header`."
        df = pd.read_csv(Path(path)/csv_name, header=header)
        return cls.from_df(df, path=path, cols=cols, **kwargs)

    def _relative_item_path(self, i): return self.items[i].relative_to(self.path)
    def _relative_item_paths(self):   return [self._relative_item_path(i) for i in range_of(self.items)]

    def use_partial_data(self, sample_pct:float=1.0, seed:int=None)->'ItemList':
        "Use only a sample of `sample_pct`of the full dataset and an optional `seed`."
        if seed is not None: np.random.seed(seed)
        rand_idx = np.random.permutation(range_of(self))
        cut = int(sample_pct * len(self))
        return self[rand_idx[:cut]]

    def to_text(self, fn:str):
        "Save `self.items` to `fn` in `self.path`."
        with open(self.path/fn, 'w') as f: f.writelines([f'{o}\n' for o in self._relative_item_paths()])

    def filter_by_func(self, func:Callable)->'ItemList':
        "Only keep elements for which `func` returns `True`."
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

    def filter_by_rand(self, p:float, seed:int=None):
        "Keep random sample of `items` with probability `p` and an optional `seed`."
        if seed is not None: np.random.seed(seed)
        return self.filter_by_func(lambda o: rand_bool(p))

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
        return [i for i in range_of(self) if self.items[i].parts[self.num_parts]==name]

    def split_by_folder(self, train:str='train', valid:str='valid')->'ItemLists':
        "Split the data depending on the folder (`train` or `valid`) in which the filenames are."
        return self.split_by_idxs(self._get_by_folder(train), self._get_by_folder(valid))

    def random_split_by_pct(self, valid_pct:float=0.2, seed:int=None)->'ItemLists':
        "Split the items randomly by putting `valid_pct` in the validation set, optional `seed` can be passed."
        if seed is not None: np.random.seed(seed)
        rand_idx = np.random.permutation(range_of(self))
        cut = int(valid_pct * len(self))
        return self.split_by_idx(rand_idx[:cut])

    def split_by_valid_func(self, func:Callable)->'ItemLists':
        "Split the data by result of `func` (which returns `True` for validation set)."
        valid_idx = [i for i,o in enumerate(self.items) if func(o)]
        return self.split_by_idx(valid_idx)

    def split_by_files(self, valid_names:'ItemList')->'ItemLists':
        "Split the data by using the names in `valid_names` for validation."
        return self.split_by_valid_func(lambda o: o.name in valid_names)

    def split_by_fname_file(self, fname:PathOrStr, path:PathOrStr=None)->'ItemLists':
        "Split the data by using the names in `fname` for the validation set. `path` will override `self.path`."
        path = Path(ifnone(path, self.path))
        valid_names = loadtxt_str(self.path/fname)
        return self.split_by_files(valid_names)

    def split_from_df(self, col:IntsOrStrs=2):
        "Split the data from the `col` in the dataframe in `self.xtra`."
        valid_idx = np.where(self.xtra.iloc[:,df_names_to_idx(col, self.xtra)])[0]
        return self.split_by_idx(valid_idx)

    def get_label_cls(self, labels, label_cls:Callable=None, sep:str=None, **kwargs):
        "Return `label_cls` or guess one from the first element of `labels`."
        if label_cls is not None:               return label_cls
        if self.label_cls is not None:          return self.label_cls
        it = index_row(labels,0)
        if sep is not None:                     return MultiCategoryList
        if isinstance(it, (float, np.float32)): return FloatList
        if isinstance(try_int(it), (str,int)):  return CategoryList
        if isinstance(it, Collection):          return MultiCategoryList
        return self.__class__

    def label_from_list(self, labels:Iterator, **kwargs)->'LabelList':
        "Label `self.items` with `labels`."
        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, **kwargs)
        y = label_cls(labels, path=self.path, **kwargs)
        res = self._label_list(x=self, y=y)
        return res

    def label_from_df(self, cols:IntsOrStrs=1, **kwargs):
        "Label `self.items` from the values in `cols` in `self.xtra`."
        labels = _maybe_squeeze(self.xtra.iloc[:,df_names_to_idx(cols, self.xtra)])
        return self.label_from_list(labels, **kwargs)

    def label_const(self, const:Any=0, **kwargs)->'LabelList':
        "Label every item with `const`."
        return self.label_from_func(func=lambda o: const, **kwargs)

    def label_empty(self):
        "Label every item with an `EmptyLabel`."
        return self.label_from_func(func=lambda o: 0., label_cls=EmptyLabelList)

    def label_from_func(self, func:Callable, **kwargs)->'LabelList':
        "Apply `func` to every input to get its label."
        return self.label_from_list([func(o) for o in self.items], **kwargs)

    def label_from_folder(self, **kwargs)->'LabelList':
        "Give a label to each filename depending on its folder."
        return self.label_from_func(func=lambda o: o.parts[-2], **kwargs)

    def label_from_re(self, pat:str, full_path:bool=False, **kwargs)->'LabelList':
        "Apply the re in `pat` to determine the label of every filename.  If `full_path`, search in the full name."
        pat = re.compile(pat)
        def _inner(o):
            s = str(os.path.join(self.path,o) if full_path else o)
            res = pat.search(s)
            assert res,f'Failed to find "{pat}" in "{s}"'
            return res.group(1)
        return self.label_from_func(_inner, **kwargs)

class EmptyLabelList(ItemList):
    "Basic `ItemList` for dummy labels."
    def get(self, i): return EmptyLabel()
    def reconstruct(self, t:Tensor, x:Tensor=None):
        if len(t.size()) == 0: return EmptyLabel()
        return self.x.reconstruct(t,x) if has_arg(self.x.reconstruct, 'x') else self.x.reconstruct(t)

class CategoryProcessor(PreProcessor):
    "Processor that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds:ItemList): self.create_classes(ds.classes)

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {v:k for k,v in enumerate(classes)}

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        return uniqueify(items)

    def process_one(self,item): return self.c2i.get(item,None)

    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i
        super().process(ds)

    def __getstate__(self): return {'classes':self.classes}
    def __setstate__(self, state:dict): self.create_classes(state['classes'])

class CategoryListBase(ItemList):
    "Basic `ItemList` for classification."
    def __init__(self, items:Iterator, classes:Collection=None,**kwargs):
        self.classes=classes
        super().__init__(items, **kwargs)

    @property
    def c(self): return len(self.classes)

    def new(self, items, classes=None, **kwargs):
        return super().new(items, classes=ifnone(classes, self.classes), **kwargs)

class CategoryList(CategoryListBase):
    "Basic `ItemList` for single classification labels."
    _processor=CategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = CrossEntropyFlat()

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return Category(o, self.classes[o])

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax()

    def reconstruct(self, t):
        return Category(t, self.classes[t])

class MultiCategoryProcessor(CategoryProcessor):
    "Processor that create `classes` from `ds.items` and handle the mapping."
    def process_one(self,item): return [self.c2i.get(o,None) for o in item]

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        classes = set()
        for c in items: classes = classes.union(set(c))
        classes = list(classes)
        classes.sort()
        return classes

class MultiCategoryList(CategoryListBase):
    "Basic `ItemList` for multi-classification labels."
    _processor=MultiCategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, sep:str=None, **kwargs):
        if sep is not None: items = array(csv.reader(items.astype(str), delimiter=sep))
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = BCEWithLogitsFlat()

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return MultiCategory(one_hot(o, self.c), [self.classes[p] for p in o], o)

    def analyze_pred(self, pred, thresh:float=0.5):
        return (pred >= thresh).float()

    def reconstruct(self, t):
        o = [i for i in range(self.c) if t[i] == 1.]
        return MultiCategory(t, [self.classes[p] for p in o], o)

class FloatList(ItemList):
    "`ItemList` suitable for storing the floats in items for regression. Will add a `log` if True"
    def __init__(self, items:Iterator, log:bool=False, **kwargs):
        super().__init__(np.array(items, dtype=np.float32), **kwargs)
        self.log = log
        self.copy_new.append('log')
        self.c = self.items.shape[1] if len(self.items.shape) > 1 else 1
        self.loss_func = MSELossFlat()

    def get(self, i):
        o = super().get(i)
        return FloatItem(log(o) if self.log else o)

    def reconstruct(self,t): return FloatItem(t.item())

class ItemLists():
    "An `ItemList` for each of `train` and `valid` (optional `test`)."
    def __init__(self, path:PathOrStr, train:ItemList, valid:ItemList, test:ItemList=None):
        self.path,self.train,self.valid,self.test = Path(path),train,valid,test
        if isinstance(self.train, LabelList): self.__class__ = LabelLists

    def __repr__(self)->str:
        return f'{self.__class__.__name__};\n\nTrain: {self.train};\n\nValid: {self.valid};\n\nTest: {self.test}'

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
        label_cls = self.train.get_label_cls(train_labels, label_cls)
        self.train = self.train._label_list(x=self.train, y=label_cls(train_labels, **kwargs))
        self.valid = self.valid._label_list(x=self.valid, y=self.train.y.new(valid_labels, **kwargs))
        self.__class__ = LabelLists
        self.process()
        return self

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: return self
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

    def transform_y(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the ys of the train and validation set."
        if not tfms: tfms=(None,None)
        self.train.transform_y(tfms[0], **kwargs)
        self.valid.transform_y(tfms[1], **kwargs)
        if self.test: self.test.transform_y(tfms[1], **kwargs)
        return self

class LabelLists(ItemLists):
    "A `LabelList` for each of `train` and `valid` (optional `test`)."
    def get_processors(self):
        "Read the default class processors if none have been set."
        procs_x,procs_y = listify(self.train.x._processor),listify(self.train.y._processor)
        xp = ifnone(self.train.x.processor, [p(ds=self.train.x) for p in procs_x])
        yp = ifnone(self.train.y.processor, [p(ds=self.train.y) for p in procs_y])
        return xp,yp

    def process(self):
        "Process the inner datasets."
        xp,yp = self.get_processors()
        for i,ds in enumerate(self.lists): ds.process(xp, yp, filter_missing_y=i==0)
        return self

    def databunch(self, path:PathOrStr=None, **kwargs)->'ImageDataBunch':
        "Create an `DataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        path = Path(ifnone(path, self.path))
        return self.x._bunch.create(self.train, self.valid, test_ds=self.test, path=path, **kwargs)

    def add_test(self, items:Iterator, label:Any=None):
        "Add test set containing `items` with an arbitrary `label`"
        # if no label passed, use label of first training item
        if label is None: label = self.train[0][1].obj
        labels = [label for _ in range_of(items)]
        if isinstance(items, ItemList): self.test = self.valid.new(items.items, labels, xtra=items.xtra)
        else: self.test = self.valid.new(items, labels)
        return self

    def add_test_folder(self, test_folder:str='test', label:Any=None):
        "Add test set containing items from `test_folder` and an arbitrary `label`."
        items = self.x.__class__.from_folder(self.path/test_folder)
        return self.add_test(items.items, label=label)

class LabelList(Dataset):
    "A list of inputs `x` and labels `y` with optional `tfms`."
    def __init__(self, x:ItemList, y:ItemList, tfms:TfmList=None, tfm_y:bool=False, **kwargs):
        self.x,self.y,self.tfm_y = x,y,tfm_y
        self.y.x = x
        self.item=None
        self.transform(tfms, **kwargs)

    def __len__(self)->int: return len(self.x) if self.item is None else 1

    @contextmanager
    def set_item(self,item):
        "For inference, will replace the dataset with one that only contains `item`."
        self.item = self.x.process_one(item)
        yield None
        self.item = None

    def __repr__(self)->str:
        x = f'{self.x}' # force this to happen first
        return f'{self.__class__.__name__}\ny: {self.y}\nx: {x}'
    def predict(self, res):
        "Delegates predict call on `res` to `self.y`."
        return self.y.predict(res)

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
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])

    def to_df(self)->None:
        "Create `pd.DataFrame` containing `items` from `self.x` and `self.y`."
        return pd.DataFrame(dict(x=self.x._relative_item_paths(), y=[str(o) for o in self.y]))

    def to_csv(self, dest:str)->None:
        "Save `self.to_df()` to a CSV file in `self.path`/`dest`."
        self.to_df().to_csv(self.path/dest, index=False)

    def export(self, fn:PathOrStr):
        "Export the minimal state and save it in `fn` to load an empty version for inference."
        state = {'x_cls':self.x.__class__, 'x_proc':self.x.processor,
                 'y_cls':self.y.__class__, 'y_proc':self.y.processor,
                 'path':self.path}
        pickle.dump(state, open(fn, 'wb'))

    @classmethod
    def load_empty(cls, fn:PathOrStr, tfms:TfmList=None, tfm_y:bool=False, **kwargs):
        "Load the sate in `fn` to create an empty `LabelList` for inference."
        state = pickle.load(open(fn, 'rb'))
        x = state['x_cls']([], path=state['path'], processor=state['x_proc'])
        y = state['y_cls']([], path=state['path'], processor=state['y_proc'])
        return cls(x, y, tfms=tfms, tfm_y=tfm_y, **kwargs).process()

    def process(self, xp=None, yp=None, filter_missing_y:bool=False):
        "Launch the processing on `self.x` and `self.y` with `xp` and `yp`."
        self.y.process(yp)
        if filter_missing_y and (getattr(self.x, 'filter_missing_y', None)):
            filt = array([o is None for o in self.y])
            if filt.sum()>0: self.x,self.y = self.x[~filt],self.y[~filt]
        self.x.process(xp)
        return self

    def transform(self, tfms:TfmList, tfm_y:bool=None, **kwargs):
        "Set the `tfms` and `tfm_y` value to be applied to the inputs and targets."
        self.tfms,self.tfmargs = tfms,kwargs
        if tfm_y is not None:  self.tfm_y,self.tfms_y,self.tfmargs_y = tfm_y,tfms,kwargs
        return self

    def transform_y(self, tfms:TfmList=None, **kwargs):
        "Set `tfms` to be applied to the targets only."
        self.tfm_y=True
        if tfms is None: self.tfms_y,self.tfmargs_y = self.tfms,{**self.tfmargs, **kwargs}
        else:            self.tfms_y,self.tfmargs_y = tfms,kwargs
        return self

@classmethod
def _databunch_load_empty(cls, path, fname:str='export.pkl', tfms:TfmList=None, tfm_y:bool=False, **kwargs):
    "Load an empty `DataBunch` from the exported file in `path/fname` with optional `tfms`."
    ds = LabelList.load_empty(path/fname, tfms=(None if tfms is None else tfms[1]), tfm_y=tfm_y, **kwargs)
    return cls.create(ds,ds,path=path)

DataBunch.load_empty = _databunch_load_empty
