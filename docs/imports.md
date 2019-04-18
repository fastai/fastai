---
title: Imports
---

## Introduction

To support interactive computing, fastai provides easy access to commonly-used external modules. A star import such as:
```
from fastai.basics import *
```
will populate the current namespace with these external modules in addition to fastai-specific functions and variables. This page documents these convenience imports, which are defined in [fastai.imports](https://github.com/fastai/fastai/blob/master/fastai/imports).

Note: since this document was manually created, it could be outdated by the time you read it. To get the up-to-date listing of imports, use:

```python
python -c 'a = set([*vars().keys(), "a"]); from fastai.basics import *; print(*sorted(set(vars().keys())-a), sep="\n")'
```

*Names in bold are modules. If an object was aliased during its import, the original name is listed in parentheses.*

| Name | Description |
|-|-|
| [**`csv`**](https://docs.python.org/3/library/csv.html) | CSV file reading and writing |
| [**`gc`**](https://docs.python.org/3/library/gc.html) | Garbage collector interface |
| [**`gzip`**](https://docs.python.org/3/library/gzip.html) | Support for gzip files |
| [**`os`**](https://docs.python.org/3/library/os.html) | Miscellaneous operating system interfaces |
| [**`pickle`**](https://docs.python.org/3/library/pickle.html) | Python object serialization |
| [**`shutil`**](https://docs.python.org/3/library/shutil.html) | High level file operations |
| [**`sys`**](https://docs.python.org/3/library/sys.html) | System-specific parameters and functions |
| [**`warnings`**](https://docs.python.org/3/library/warnings.html), [`warn`](https://docs.python.org/3/library/warnings.html#warnings.warn) | Warning control |
| [**`yaml`**](https://pyyaml.org/wiki/PyYAMLDocumentation) | YAML parser and emitter |
| [**`io`**](https://docs.python.org/3/library/io.html), [`BufferedWriter`](https://docs.python.org/3/library/io.html#io.BufferedWriter), [`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) | Core tools for working with streams |
| [**`subprocess`**](https://docs.python.org/3/library/subprocess.html) | Subprocess management |
| [**`math`**](https://docs.python.org/3/library/math.html) | Mathematical functions |
| [**`plt`** (`matplotlib.pyplot`)](https://matplotlib.org/api/pyplot_api.html) | MATLAB-like plotting framework |
| [**`np`** (`numpy`)](https://www.numpy.org/devdocs/reference/index.html) , [`array`](https://www.numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array), [`cos`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html), [`exp`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html),<br/> [`log`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html), [`sin`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html), [`tan`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tan.html), [`tanh`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html) | Multi-dimensional arrays, mathematical functions |
| [**`pd`** (`pandas`)](http://pandas.pydata.org/pandas-docs/stable/), [`Series`](http://pandas.pydata.org/pandas-docs/stable/reference/series.html), [`DataFrame`](http://pandas.pydata.org/pandas-docs/stable/reference/frame.html) | Data structures and tools for data analysis |
| [**`random`**](https://docs.python.org/3/library/random.html) | Generate pseudo-random numbers |
| [**`scipy.stats`**](https://docs.scipy.org/doc/scipy/reference/stats.html) | Statistical functions |
| [**`scipy.special`**](https://docs.scipy.org/doc/scipy/reference/special.html) | Special functions |
| [`abstractmethod`](https://docs.python.org/3/library/abc.html#abc.abstractmethod), [`abstractproperty`](https://docs.python.org/3/library/abc.html#abc.abstractproperty) | Abstract base classes |
| [**`collections`**](https://docs.python.org/3/library/collections.html), [`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter), [`defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict), <br/>[`namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple), [`OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict) | Container datatypes |
| [**`abc`** (`collections.abc`)](https://docs.python.org/3/library/collections.abc.html#module-collections.abc), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable) | Abstract base classes for containers |
| [**`hashlib`**](https://docs.python.org/3/library/hashlib.html) | Secure hashes and message digests |
| [**`itertools`**](https://docs.python.org/3/library/itertools.html) | Functions creating iterators for efficient looping |
| [**`json`**](https://docs.python.org/3/library/json.html) | JSON encoder and decoder |
| [**`operator`**](https://docs.python.org/3/library/operator.html), [`attrgetter`](https://docs.python.org/3/library/operator.html#operator.attrgetter), [`itemgetter`](https://docs.python.org/3/library/operator.html#operator.itemgetter) | Standard operators as functions |
| [**`pathlib`**](https://docs.python.org/3/library/pathlib.html), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | Object-oriented filesystem paths |
| [**`mimetypes`**](https://docs.python.org/3/library/mimetypes.html) | Map filenames to MIME types |
| [**`inspect`**](https://docs.python.org/3/library/inspect.html) | Inspect live objects |
| [**`typing`**](https://docs.python.org/3/library/typing.html), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`AnyStr`](https://docs.python.org/3/library/typing.html#typing.AnyStr), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable),<br/> [`Collection`](https://docs.python.org/3/library/typing.html#typing.Collection), [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict), [`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable), [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator),<br/> [`List`](https://docs.python.org/3/library/typing.html#typing.List), [`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping), [`NewType`](https://docs.python.org/3/library/typing.html#typing.NewType), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional),<br/> [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence), [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar), [`Union`](https://docs.python.org/3/library/typing.html#typing.Union) | Support for type hints |
| [**`functools`**](https://docs.python.org/3/library/functools.html), [`partial`](https://docs.python.org/3/library/functools.html#functools.partial), [`reduce`](https://docs.python.org/3/library/functools.html#functools.reduce) | Higher-order functions and operations on callable objects |
| [**`importlib`**](https://docs.python.org/3/library/importlib.html) | The implementatin of import |
| [**`weakref`**](https://docs.python.org/3/library/weakref.html) | Weak references |
| [**`html`**](https://docs.python.org/3/library/html.html) | HyperText Markup Language support |
| [**`re`**](https://docs.python.org/3/library/re.html) | Regular expression operations |
| [**`requests`**](http://docs.python-requests.org/en/master/) | HTTP for Humans&trade; |
| [**`tarfile`**](https://docs.python.org/3/library/tarfile.html) | Read and write tar archive files |
| [**`numbers`**](https://docs.python.org/3/library/numbers.html), [`Number`](https://docs.python.org/3/library/numbers.html#numbers.Number) | Numeric abstract base classes |
| [**`tempfile`**](https://docs.python.org/3/library/tempfile.html) | Generate temporary files and directories |
| [**`concurrent`**](https://docs.python.org/3/library/concurrent.html), [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor),<br/> [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) | Launch parallel tasks |
| [`copy`](https://docs.python.org/3/library/copy.html#copy.copy), [`deepcopy`](https://docs.python.org/3/library/copy.html#copy.deepcopy) | Shallow and deep copy operation |
| [`dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass), [`field`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field), `InitVar` | Data Classes |
| [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum), [`IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum) | Support for enumerations |
| [`set_trace`](https://docs.python.org/3/library/pdb.html#pdb.set_trace) | The Python debugger |
| [**`patches`** (`matplotlib.patches`)](https://matplotlib.org/api/patches_api.html), [`Patch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch) | ? |
| [**`patheffects`** (`matplotlib.patheffects`)](https://matplotlib.org/api/patheffects_api.html) | ? |
| [`contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager) | Utilities for `with`-statement contexts |
| [`MasterBar`, `master_bar`, `ProgressBar`,<br/> `progress_bar`](https://github.com/fastai/fastprogress) | Simple and flexible progress bar for Jupyter Notebook and console |
| [**`pkg_resources`**](https://setuptools.readthedocs.io/en/latest/pkg_resources.html) | Package discovery and resource access |
| [`SimpleNamespace`](https://docs.python.org/3/library/types.html#types.SimpleNamespace) | Dynamic type creation and names for built-in types |
| [**`torch`**](https://pytorch.org/docs/stable/), [`as_tensor`](https://pytorch.org/docs/stable/torch.html?highlight=as_tensor#torch.as_tensor), [`ByteTensor`](https://pytorch.org/docs/stable/tensors.html#torch.ByteTensor),<br/> [`DoubleTensor`, `FloatTensor`, `HalfTensor`,<br/> `LongTensor`, `ShortTensor`, `Tensor`](https://pytorch.org/docs/stable/tensors.html) | Tensor computation and deep learning |
| [**`nn`** (`torch.nn`)](https://pytorch.org/docs/stable/nn.html), [`weight_norm`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.weight_norm), [`spectral_norm`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.spectral_norm) | Neural networks with PyTorch |
| [**`F`** (`torch.nn.functional`)](https://pytorch.org/docs/stable/nn.html#torch-nn-functional) | PyTorch functional interface |
| [**`optim`** (`torch.optim`)](https://pytorch.org/docs/stable/optim.html) | Optimization algorithms in PyTorch |
| [`BatchSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler), [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset),<br/> [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler), [`TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) | PyTorch data utils |
