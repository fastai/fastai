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

_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus, cmap='viridis')

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def is_tuple(x:Any)->bool: return isinstance(x, tuple)
def noop(x): return x

def chunks(l:Collection, n:int)->Iterable:
    "Yield successive `n`-sized chunks from `l`."
    for i in range(0, len(l), n): yield l[i:i+n]

def to_int(b:Any)->Union[int,List[int]]:
    "Convert `b` to an int or list of ints (if `is_listy`); raises exception if not convertible"
    if is_listy(b): return [to_int(x) for x in b]
    else:          return int(b)

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def is1d(a:Collection)->bool:
    "Return `True` if `a` is one-dimensional"
    return len(a.shape) == 1 if hasattr(a, 'shape') else True

def uniqueify(x:Series)->List:
    "Return sorted unique values of `x`."
    res = list(OrderedDict.fromkeys(x).keys())
    res.sort()
    return res

def idx_dict(a): 
    "Create a dictionary value to index from `a`."
    return {v:k for k,v in enumerate(a)}

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
    "Make `p` listy and the same length as `q`."
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
    "Change `name` from camel to snake style."
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def even_mults(start:float, stop:float, n:int)->np.ndarray:
    "Build log-stepped array from `start` to `stop` in `n` steps."
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
    "Base item type in the fastai library."
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

def range_of(x):  
    "Create a range from 0 to `len(x)`."
    return list(range(len(x)))
def arange_of(x): 
    "Same as `range_of` but returns an array."
    return np.arange(len(x))

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
    "One-hot encode `x` with `c` classes."
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

def has_arg(func, arg)->bool: 
    "Check if `func` accepts `arg`."
    return arg in func_args(func)

def split_kwargs_by_func(kwargs, func):
    "Split `kwargs` between those expected by `func` and the others."
    args = func_args(func)
    func_kwargs = {a:kwargs.pop(a) for a in args if a in kwargs}
    return func_kwargs, kwargs

def try_int(o:Any)->Any:
    "Try to convert `o` to int, default to `o` if not possible."
    try: return int(o)
    except: return o

def array(a, *args, **kwargs)->np.ndarray:
    "Same as `np.array` but also handles generators"
    if not isinstance(a, collections.Sized) and not getattr(a,'__array_interface__',False):
        a = list(a)
    return np.array(a, *args, **kwargs)

class EmptyLabel(ItemBase):
    "Should be used for a dummy label."
    def __init__(self): self.obj,self.data = 0.,0.
    def __str__(self):  return ''

class Category(ItemBase):
    "Basic class for singe classification labels."
    def __init__(self,data,obj): self.data,self.obj = data,obj
    def __int__(self): return int(self.data)
    def __str__(self): return str(self.obj)

class MultiCategory(ItemBase):
    "Basic class for multi-classification labels."
    def __init__(self,data,obj,raw): self.data,self.obj,self.raw = data,obj,raw
    def __str__(self): return ';'.join([str(o) for o in self.obj])

def _treat_html(o:str)->str:
    return o.replace('\n','\\n')

def text2html_table(items:Collection[Collection[str]], widths:Collection[int])->str:
    "Put the texts in `items` in an HTML table, `widths` are the widths of the columns in %."
    html_code = f"<table>"
    for w in widths: html_code += f"  <col width='{w}%'>"
    for line in items:
        html_code += "  <tr>\n"
        html_code += "\n".join([f"    <th>{_treat_html(o)}</th>" for o in line if len(o) >= 1])
        html_code += "\n  </tr>\n"
    return html_code + "</table>\n"

def parallel(func, arr:Collection, max_workers:int=None):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers<2: _ = [func(o,i) for i,o in enumerate(arr)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr)): pass

def subplots(rows:int, cols:int, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, title=None, **kwargs):
    "Like `plt.subplots` but with consistent axs shape, `kwargs` passed to `fig.suptitle` with `title`"
    figsize = ifnone(figsize, (imgsize*cols, imgsize*rows))
    fig, axs = plt.subplots(rows,cols,figsize=figsize)
    if (rows==1 and cols!=1) or (cols==1 and rows!=1): axs = [axs]
    if title is not None: fig.suptitle(title, **kwargs)
    return array(axs)

