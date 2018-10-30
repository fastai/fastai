from .torch_core import *
from .basic_data import *

__all__ = ['InputList', 'ItemList', 'LabelList', 'PathItemList', 'SplitData', 'SplitDatasets', 'get_files']

def get_files(c:PathOrStr, extensions:Collection[str]=None, recurse:bool=False)->FilePathList:
    "Return list of files in `c` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
    return [o for o in Path(c).glob('**/*' if recurse else '*')
            if not o.name.startswith('.') and not o.is_dir()
            and (extensions is None or (o.suffix.lower() in extensions))]

class ItemList():
    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    def __init__(self, items:Iterator): self.items = np.array(list(items))
    def __len__(self)->int: return len(self.items)
    def __getitem__(self,i:int)->Any: self.items[i]
    def __repr__(self)->str: return f'{self.__class__.__name__} ({len(self)} items)\n{self.items}'

class PathItemList(ItemList):
    "An `ItemList` with a `path` attribute."
    def __init__(self, items:Iterator, path:PathOrStr='.'):
        super().__init__(items)
        self.path = Path(path)
    def __repr__(self)->str: return f'{super().__repr__()}\nPath: {self.path}'

def _df_to_fns_labels(df:pd.DataFrame, fn_col:int=0, label_col:int=1,
                      label_delim:str=None, suffix:Optional[str]=None):
    """Get image file names in `fn_col` by adding `suffix` and labels in `label_col` from `df`.
    If `label_delim` is specified, splits the values in `label_col` accordingly.
    """
    if label_delim:
        df.iloc[:,label_col] = list(csv.reader(df.iloc[:,label_col], delimiter=label_delim))
    labels = df.iloc[:,label_col].values
    fnames = df.iloc[:,fn_col].str.lstrip()
    if suffix: fnames = fnames + suffix
    return fnames.values, labels

class InputList(PathItemList):
    "A list of inputs. Contain methods to get the corresponding labels."
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, recurse=True)->'InputList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions=extensions, recurse=recurse), path)

    def label_from_func(self, func:Callable)->'LabelList':
        "Apply `func` to every input to get its label."
        return LabelList([(o,func(o)) for o in self.items], self.path)

    def label_from_re(self, pat:str, full_path:bool=False)->'LabelList':
        """Apply the re in `pat` to determine the label of every filename.  If `full_path`, search in the full name.
        This method is primarly intended for inputs that are filenames, but could work in other settings."""
        pat = re.compile(pat)
        def _inner(o):
            s = str(o if full_path else o.name)
            res = pat.search(s)
            assert res,f'Failed to find "{pat}" in "{s}"'
            return res.group(1)
        return self.label_from_func(_inner)

    def label_from_df(self, df, fn_col:int=0, label_col:int=1, sep:str=None, folder:PathOrStr='.',
                      suffix:str=None)->'LabelList':
        """Look in `df` for the filenames in `fn_col` to get the corresponding label in `label_col`.
        If a `folder` is specified, filenames are taken in `self.path/folder`. `suffix` is added.
        If `sep` is specified, splits the values in `label_col` accordingly.
        This method is intended for inputs that are filenames."""
        fnames, labels = _df_to_fns_labels(df, fn_col, label_col, sep, suffix)
        fnames = join_paths(fnames, self.path/Path(folder))
        return LabelList([(fn, np.array(lbl, dtype=np.object)) for fn, lbl in zip(fnames, labels) if fn in self.items],
                         self.path)

    def label_from_csv(self, csv_fname, header:Optional[Union[int,str]]='infer', fn_col:int=0, label_col:int=1,
                       sep:str=None, folder:PathOrStr='.', suffix:str=None)->'LabelList':
        """Look in `self.path/csv_fname` for a csv loaded with an optional `header` containing the filenames in
        `fn_col` to get the corresponding label in `label_col`.
        If a `folder` is specified, filenames are taken in `path/folder`. `suffix` is added.
        If `sep` is specified, splits the values in `label_col` accordingly.
        This method is intended for inputs that are filenames."""
        df = pd.read_csv(self.path/csv_fname, header=header)
        return self.label_from_df(df, fn_col, label_col, sep, folder, suffix)

    def label_from_folder(self, classes:Collection[str]=None)->'LabelList':
        """Give a label to each filename depending on its folder. If `classes` are specified, only keep those.
        This method is intended for inputs that are filenames."""
        labels = [fn.parent.parts[-1] for fn in self.items]
        if classes is None: classes = uniqueify(labels)
        return LabelList([(o,lbl) for o, lbl in zip(self.items, labels) if lbl in classes], self.path)

class LabelList(PathItemList):
    "A list of inputs and labels. Contain methods to split it in `SplitData`."
    @property
    def files(self): return self.items[:,0]

    def split_by_files(self, valid_names:InputList)->'SplitData':
        "Split the data by using the names in `valid_names` for validation."
        valid = [o for o in self.items if o[0] in valid_names]
        train = [o for o in self.items if o[0] not in valid_names]
        return SplitData(self.path, LabelList(train), LabelList(valid))

    def split_by_fname_file(self, fname:PathOrStr, path:PathOrStr=None)->'SplitData':
        """Split the data by using the file names in `fname` for the validation set. `path` will override `self.path`.
        This method won't work if you inputs aren't filenames.
        """
        path = Path(ifnone(path, self.path))
        valid_names = join_paths(loadtxt_str(self.path/fname), path)
        return self.split_by_files(valid_names)

    def split_by_idx(self, valid_idx:Collection[int])->'SplitData':
        "Split the data according to the indexes in `valid_idx`."
        valid = [o for i,o in enumerate(self.items) if i in valid_idx]
        train = [o for i,o in enumerate(self.items) if i not in valid_idx]
        return SplitData(self.path, LabelList(train), LabelList(valid))

    def split_by_folder(self, train:str='train', valid:str='valid')->'SplitData':
        """Split the data depending on the folder (`train` or `valid`) in which the filenames are.
        This method won't work if you inputs aren't filenames.
        """
        n = len(self.path.parts)
        folder_name = [o[0].parent.parts[n] for o in self.items]
        valid = [o for o in self.items if o[0].parent.parts[n] == valid]
        train = [o for o in self.items if o[0].parent.parts[n] == train]
        return SplitData(self.path, LabelList(train), LabelList(valid))

    def random_split_by_pct(self, valid_pct:float=0.2)->'SplitData':
        "Split the items randomly by putting `valid_pct` in the validation set."
        rand_idx = np.random.permutation(range(len(self.items)))
        cut = int(valid_pct * len(self.items))
        return self.split_by_idx(rand_idx[:cut])

@dataclass
class SplitData():
    "Regroups `train` and `valid` data, inside a `path`."
    path:PathOrStr
    train:LabelList
    valid:LabelList

    def __post_init__(self): self.path = Path(self.path)

    @property
    def lists(self): return [self.train,self.valid]

    def datasets(self, dataset_cls:type, **kwargs)->'SplitDatasets':
        "Create datasets from the underlying data using `dataset_cls` and passing the `kwargs`."
        dss = [dataset_cls(*self.train.items.T, **kwargs)]
        kwg_cls = kwargs.pop('classes') if 'classes' in kwargs else None
        if hasattr(dss[0], 'classes'): kwg_cls = dss[0].classes
        if kwg_cls is not None: kwargs['classes'] = kwg_cls
        dss.append(dataset_cls(*self.valid.items.T, **kwargs))
        cls = getattr(dataset_cls, '__splits_class__', SplitDatasets)
        return cls(self.path, *dss)

@dataclass
class SplitDatasets():
    "A class regrouping `train_ds`, a `valid_ds` and maybe a `train_ds` dataset, inside a `path`."
    path:PathOrStr
    train_ds:Dataset
    valid_ds:Dataset
    test_ds:Optional[Dataset] = None

    @property
    def datasets(self)->Collection[Dataset]:
        "The underlying datasets of this object."
        return [self.train_ds,self.valid_ds] if self.test_ds is None else [self.train_ds,self.valid_ds, self.test_ds]

    def dataloaders(self, **kwargs)->Collection[DataLoader]:
        "Create dataloaders with the inner datasets, pasing the `kwargs`."
        return [DataLoader(o, **kwargs) for o in self.datasets]

    @classmethod
    def from_single(cls, path:PathOrStr, ds:Dataset)->'SplitDatasets':
        "Factory method that uses `ds` for both valid and train, and passes `path`."
        return cls(path, ds, ds)

    @classmethod
    def single_from_classes(cls, path:PathOrStr, classes:Collection[str])->'SplitDatasets':
        "Factory method that passes a `SingleClassificationDataset` on `classes` to `from_single`."
        return cls.from_single(path, SingleClassificationDataset(classes))

    @classmethod
    def single_from_c(cls, path:PathOrStr, c:int)->'SplitDatasets':
        "Factory method that passes a `DatasetBase` on `c` to `from_single`."
        return cls.from_single(path, DatasetBase(c))

