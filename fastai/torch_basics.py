from torch import multiprocessing
# Python 3.8 changed to 'spawn' but that doesn't work with PyTorch DataLoader w n_workers>0
import platform
if platform.system()=='Darwin': multiprocessing.set_start_method('fork', force=True)

from .imports import *
from .torch_imports import *
from .torch_core import *
from .layers import *
