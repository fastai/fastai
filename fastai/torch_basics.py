from torch import multiprocessing
import platform,os
if platform.system()=='Darwin':
    # Python 3.8 changed to 'spawn' but that doesn't work with PyTorch DataLoader w n_workers>0
    multiprocessing.set_start_method('fork', force=True)
    # workaround "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from .imports import *
from .torch_imports import *
from .torch_core import *
from .layers import *
from .losses import *
