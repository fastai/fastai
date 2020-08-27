from torch import multiprocessing
if platform.system()=='Darwin': multiprocessing.set_start_method('fork', force=True)
from .imports import *
from .torch_imports import *
from .torch_core import *
from .layers import *
