"Utility functions for memory management"

from ..imports.torch import *
from ..core import *
from ..script import *
import pynvml, functools, traceback
from collections import namedtuple

GPUMemory = namedtuple('GPUMemory', ['total', 'used', 'free'])

have_cuda = 0
if torch.cuda.is_available():
    pynvml.nvmlInit()
    have_cuda = 1

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)

# for gpu returns GPUMemory(total, used, free)
# for cpu returns GPUMemory(0, 0, 0)
# for invalid gpu id returns GPUMemory(0, 0, 0)
def gpu_mem_get(id=None):
    "query nvidia for total, used and free memory for gpu in MBs. if id is not passed, currently selected torch device is used"
    if not have_cuda: return GPUMemory(0, 0, 0)
    if id is None: id = torch.cuda.current_device()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMemory(*(map(b2mb, [info.total, info.used, info.free])))
    except:
        return GPUMemory(0, 0, 0)

# for gpu returns [ GPUMemory(total_0, used_0, free_0), GPUMemory(total_1, used_1, free_1), .... ]
# for cpu returns []
def gpu_mem_get_all():
    "query nvidia for total, used and free memory for each available gpu in MBs"
    if not have_cuda: return []
    return list(map(gpu_mem_get, range(pynvml.nvmlDeviceGetCount())))

# for gpu returns: (gpu_with_max_free_ram_id, its_free_ram)
# for cpu returns: (None, 0)
def gpu_with_max_free_mem():
    "get [gpu_id, its_free_ram] for the first gpu with highest available RAM"
    mem_all = gpu_mem_get_all()
    if not len(mem_all): return None, 0
    free_all = np.array([x.free for x in mem_all])
    id = np.argmax(free_all)
    return id, free_all[id]

def get_ref_free_exc_info():
    "Free traceback from references to locals() in each frame to avoid circular reference leading to gc.collect() unable to reclaim memory"
    type, val, tb = sys.exc_info()
    traceback.clear_frames(tb)
    return (type, val, tb)

# this is a decorator to be used with any functions that interact with CUDA (top-level is fine)
#
# ipython has a bug where it stores tb with all the locals() tied in (circular
# reference) and are unable to be freed, so we cleanse the tb before handing it
# over to ipython.
def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = get_ref_free_exc_info() # must!
            raise type(val).with_traceback(tb) from None
    return wrapper
