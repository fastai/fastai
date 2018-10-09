"`fastai.core` contains essential util functions to format and split data"
from .imports.core import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

AnnealFunc = Callable[[Number,Number,float], Number]
ArgStar = Collection[Any]
BatchSamples = Collection[Tuple[Collection[int], int]]
Classes = Collection[Any]
DataFrameOrChunks = Union[DataFrame, pd.io.parsers.TextFileReader]
FilePathList = Collection[Path]
Floats = Union[float, Collection[float]]
ImgLabel = str
ImgLabels = Collection[ImgLabel]
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

def num_cpus()->int:
    "Get number of cpus"
    try:                   return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count()

default_cpus = min(16, num_cpus())

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def is_tuple(x:Any)->bool: return isinstance(x, tuple)
def noop(x): return x

def to_int(b):
    if is_listy(b): return [to_int(x) for x in b]
    else:          return int(b)

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def uniqueify(x:Series) -> List[Any]: return list(OrderedDict.fromkeys(x).keys())
def idx_dict(a): return {v:k for k,v in enumerate(a)}

def find_classes(folder:Path)->FilePathList:
    "List of label subdirectories in imagenet-style `folder`."
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    assert(len(classes)>0)
    return sorted(classes, key=lambda d: d.name)

def arrays_split(mask:NPArrayMask, *arrs:NPArrayableList)->SplitArrayList:
    "Given `arrs` is [a,b,...] and `mask`index - return[(a[mask],a[~mask]),(b[mask],b[~mask]),...]."
    mask = array(mask)
    return list(zip(*[(a[mask],a[~mask]) for a in map(np.array, arrs)]))

def random_split(valid_pct:float, *arrs:NPArrayableList)->SplitArrayList:
    "Randomly split `arrs` with `valid_pct` ratio. good for creating validation set."
    is_train = np.random.uniform(size=(len(arrs[0]),)) > valid_pct
    return arrays_split(is_train, *arrs)

def listify(p:OptListOrItem=None, q:OptListOrItem=None):
    "Make `p` same length as `q`"
    if p is None: p=[]
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
    "Extracs the keys in `names` from the `kwargs`."
    new_kwargs = {}
    for arg_name in names:
        if arg_name in kwargs:
            arg_val = kwargs.pop(arg_name)
            new_kwargs[arg_name] = arg_val
    return new_kwargs, kwargs

def partition(a:Collection, sz:int) -> List[Collection]:
    "Split iterables `a` in equal parts of size `sz`"
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a:Collection, n_cpus:int) -> List[Collection]:
    "Split data in `a` equally among `n_cpus` cores"
    return partition(a, len(a)//n_cpus + 1)

def get_chunk_length(data:Union[PathOrStr, DataFrame, pd.io.parsers.TextFileReader], chunksize:Optional[int] = None) -> int:
    "Read the number of chunks in a pandas `DataFrame`."
    if (type(data) == DataFrame):  return 1
    elif (type(data) == pd.io.parsers.TextFileReader):
        dfs = pd.read_csv(data.f, header=None, chunksize=data.chunksize)
    else:  dfs = pd.read_csv(data, header=None, chunksize=chunksize)
    l = 0
    for _ in dfs: l+=1
    return l

def get_total_length(csv_name:PathOrStr, chunksize:int) -> int:
    "Read the the total length of a pandas `DataFrame`."
    dfs = pd.read_csv(csv_name, header=None, chunksize=chunksize)
    l = 0
    for df in dfs: l+=len(df)
    return l

def maybe_copy(old_fnames:Collection[PathOrStr], new_fnames:Collection[PathOrStr]):
    "Copy the `old_fnames` to `new_fnames` location if `new_fnames` don't exist or are less recent."
    os.makedirs(os.path.dirname(new_fnames[0]), exist_ok=True)
    for old_fname,new_fname in zip(old_fnames, new_fnames):
        if not os.path.isfile(new_fname) or os.path.getmtime(new_fname) < os.path.getmtime(old_fname):
            shutil.copyfile(old_fname, new_fname)

def series2cat(df:DataFrame, *col_names):
    "Categorifies the columns `col_names` in `df`."
    for c in listify(col_names): df[c] = df[c].astype('category').cat.as_ordered()

class ItemBase():
    "All transformable dataset items use this type."
    @property
    @abstractmethod
    def device(self): pass
    @property
    @abstractmethod
    def data(self): pass

def download_url(url:str, dest:str, overwrite:bool=False)->None:
    "Download `url` to `dest` unless is exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return
    u = requests.get(url, stream=True)
    file_size = int(u.headers["Content-Length"])
    u = u.raw

    with open(dest,'wb') as f:
        pbar = progress_bar(range(file_size), auto_update=False)
        nbytes,buffer = 0,[1]
        while len(buffer):
            buffer = u.read(8192)
            nbytes += len(buffer)
            pbar.update(nbytes)
            f.write(buffer)

def range_of(x): return list(range(len(x)))
def arange_of(x): return np.arange(len(x))