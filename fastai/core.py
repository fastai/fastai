"`fastai.core` contains essential util functions to format and split data"
from .imports.core import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

AnnealFunc = Callable[[Number,Number,float], Number]
ArgStar = Collection[Any]
BatchSamples = Collection[Tuple[Collection[int], int]]
DataFrameOrChunks = Union[DataFrame, pd.io.parsers.TextFileReader]
FilePathList = Collection[Path]
Floats = Union[float, Collection[float]]
ImgLabel = str
ImgLabels = Collection[ImgLabel]
IntsOrStrs = Union[int, Collection[int], str, Collection[str]]
KeyFunc = Callable[[int], int]
KWArgs = Dict[str,Any]
ListOrItem = Union[Collection[Any],int,float,str]
ListRules = Collection[Callable[[str],str]]
ListSizes = Collection[Tuple[int,int]]
NPArrayableList = Collection[Union[np.ndarray, list]]
NPArrayList = Collection[np.ndarray]
NPArrayMask = np.ndarray
NPImage = np.ndarray
OptDataFrame = Optional[DataFrame]
OptListOrItem = Optional[ListOrItem]
OptRange = Optional[Tuple[float,float]]
OptStrTuple = Optional[Tuple[str,str]]
OptStats = Optional[Tuple[np.ndarray, np.ndarray]]
PathOrStr = Union[Path,str]
PBar = Union[MasterBar, ProgressBar]
Point=Tuple[float,float]
Points=Collection[Point]
Sizes = List[List[int]]
SplitArrayList = List[Tuple[np.ndarray,np.ndarray]]
StartOptEnd=Union[float,Tuple[float,float]]
StrList = Collection[str]
Tokens = Collection[Collection[str]]
OptStrList = Optional[StrList]

np.set_printoptions(precision=6, threshold=50, edgeitems=4, linewidth=120)

def num_cpus()->int:
    "Get number of cpus"
    try:                   return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count()

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def is_tuple(x:Any)->bool: return isinstance(x, tuple)
def noop(x): return x

def to_int(b:Any)->Union[int,List[int]]:
    "Convert `b` to an int or list of ints (if `is_listy`); raises exception if not convertible"
    if is_listy(b): return [to_int(x) for x in b]
    else:          return int(b)

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def is1d(a:Collection)->bool:
    "Returns True if a collection is one dimensional"
    return len(a.shape) == 1 if hasattr(a, 'shape') else True

def uniqueify(x:Series)->List:
    "Return unique values of `x`"
    return list(OrderedDict.fromkeys(x).keys())

def idx_dict(a): return {v:k for k,v in enumerate(a)}

def find_classes(folder:Path)->FilePathList:
    "List of label subdirectories in imagenet-style `folder`."
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def arrays_split(mask:NPArrayMask, *arrs:NPArrayableList)->SplitArrayList:
    "Given `arrs` is [a,b,...] and `mask`index - return[(a[mask],a[~mask]),(b[mask],b[~mask]),...]."
    assert all([len(arr)==len(arrs[0]) for arr in arrs]), 'All arrays should have same length'
    mask = array(mask)
    return list(zip(*[(a[mask],a[~mask]) for a in map(np.array, arrs)]))

def random_split(valid_pct:float, *arrs:NPArrayableList)->SplitArrayList:
    "Randomly split `arrs` with `valid_pct` ratio. good for creating validation set."
    assert (valid_pct>=0 and valid_pct<=1), 'Validation set percentage should be between 0 and 1'
    is_train = np.random.uniform(size=(len(arrs[0]),)) > valid_pct
    return arrays_split(is_train, *arrs)

def listify(p:OptListOrItem=None, q:OptListOrItem=None):
    "Make `p` same length as `q`"
    if p is None: p=[]
    elif isinstance(p, str):          p=[p]
    elif not isinstance(p, Iterable): p=[p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def even_mults(start:float, stop:float, n:int)->np.ndarray:
    "Build evenly stepped schedule from `start` to `stop` in `n` steps."
    mult = stop/start
    step = mult**(1/(n-1))
    return np.array([start*(step**i) for i in range(n)])

def extract_kwargs(names:Collection[str], kwargs:KWArgs):
    "Extract the keys in `names` from the `kwargs`."
    new_kwargs = {}
    for arg_name in names:
        if arg_name in kwargs:
            arg_val = kwargs.pop(arg_name)
            new_kwargs[arg_name] = arg_val
    return new_kwargs, kwargs

def partition(a:Collection, sz:int)->List[Collection]:
    "Split iterables `a` in equal parts of size `sz`"
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a:Collection, n_cpus:int)->List[Collection]:
    "Split data in `a` equally among `n_cpus` cores"
    return partition(a, len(a)//n_cpus + 1)

def series2cat(df:DataFrame, *col_names):
    "Categorifies the columns `col_names` in `df`."
    for c in listify(col_names): df[c] = df[c].astype('category').cat.as_ordered()

TfmList = Union[Callable, Collection[Callable]]

class ItemBase():
    "All transformable dataset items use this type."
    def __init__(self, data:Any): self.data=self.obj=data
    def __repr__(self): return f'{self.__class__.__name__} {self}'
    def show(self, ax:plt.Axes, **kwargs): ax.set_title(str(self))
    def apply_tfms(self, tfms:Collection, **kwargs):
        if tfms: raise Exception('Not implemented')
        return self
def download_url(url:str, dest:str, overwrite:bool=False, pbar:ProgressBar=None,
                 show_progress=True, chunk_size=1024*1024, timeout=4)->None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return

    u = requests.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress: pbar = progress_bar(range(file_size), auto_update=False, leave=False, parent=pbar)
        for chunk in u.iter_content(chunk_size=chunk_size):
            nbytes += len(chunk)
            if show_progress: pbar.update(nbytes)
            f.write(chunk)

def range_of(x): return list(range(len(x)))
def arange_of(x): return np.arange(len(x))

Path.ls = lambda x: list(x.iterdir())

def join_path(fname:PathOrStr, path:PathOrStr='.')->Path:
    "Return `Path(path)/Path(fname)`, `path` defaults to current dir."
    return Path(path)/Path(fname)

def join_paths(fnames:FilePathList, path:PathOrStr='.')->Collection[Path]:
    "Join `path` to every file name in `fnames`."
    path = Path(path)
    return [join_path(o,path) for o in fnames]

def loadtxt_str(path:PathOrStr)->np.ndarray:
    "Return `ndarray` of `str` of lines of text from `path`."
    with open(path, 'r') as f: lines = f.readlines()
    return np.array([l.strip() for l in lines])

def save_texts(fname:PathOrStr, texts:Collection[str]):
    "Save in `fname` the content of `texts`."
    with open(fname, 'w') as f:
        for t in texts: f.write(f'{t}\n')

def df_names_to_idx(names:IntsOrStrs, df:DataFrame):
    "Return the column indexes of `names` in `df`."
    if not is_listy(names): names = [names]
    if isinstance(names[0], int): return names
    return [df.columns.get_loc(c) for c in names]

def one_hot(x:Collection[int], c:int):
    "One-hot encode the target."
    res = np.zeros((c,), np.float32)
    res[x] = 1.
    return res

def index_row(a:Union[Collection,pd.DataFrame,pd.Series], idxs:Collection[int])->Any:
    "Return the slice of `a` corresponding to `idxs`."
    if a is None: return a
    if isinstance(a,(pd.DataFrame,pd.Series)):
        res = a.iloc[idxs]
        if isinstance(res,(pd.DataFrame,pd.Series)): return res.copy()
        return res
    return a[idxs]

def func_args(func)->bool:
    "Return the arguments of `func`."
    code = func.__code__
    return code.co_varnames[:code.co_argcount]

def has_arg(func, arg)->bool: return arg in func_args(func)

def try_int(o:Any)->Any:
    "Try to conver `o` to int, default to `o` if not possible."
    try: return int(o)
    except: return o

def array(a, *args, **kwargs)->np.ndarray:
    "Same as `np.array` but also handles generators"
    if not isinstance(a, collections.Sized): a = list(a)
    return np.array(a, *args, **kwargs)

class Category(ItemBase):
    def __init__(self,data,obj): self.data,self.obj = data,obj
    def __int__(self): return int(self.data)
    def __str__(self): return str(self.obj)

class MultiCategory(ItemBase):
    def __init__(self,data,obj,raw): self.data,self.obj,self.raw = data,obj,raw
    def __str__(self): return ';'.join([str(o) for o in self.obj])
