"Utility functions for memory management"

# python -c "import fastai.utils.mem; print(fastai.utils.mem.get_gpu_mem(), fastai.utils.mem.get_gpu_mem_all(), fastai.utils.mem.get_gpu_with_max_free_mem())"

from ..imports.torch import *
from ..core import *
from ..script import *
import pynvml

from enum import IntEnum
Memory = IntEnum('Memory', "TOTAL, USED, FREE", start=0)

have_cuda = 0
if torch.cuda.is_available():
    pynvml.nvmlInit()
    have_cuda = 1

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)

# for gpu returns [total-0, used-0, free-0]
# for cpu returns [0, 0, 0]
# for invalid id returns [0, 0, 0]
def gpu_mem_get(id=None):
    "query nvidia for total, used and free memory for gpu in MBs. if id is not passed, currently selected torch device is used"
    if not have_cuda: return [0, 0, 0]
    if id is None: id = torch.cuda.current_device()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return [b2mb(x) for x in [info.total, info.used, info.free]]
    except:
        return [0, 0, 0]

# for gpu returns [ [total-0, used-0, free-0], [total-1, used-1, free-1] ]
# for cpu returns []
def gpu_mem_get_all():
    "query nvidia for total, used and free memory for each available gpu in MBs"
    if not have_cuda: return []
    #return []
    #return [[8119, 2969, 5150],[18119, 12969, 15150] ]
    return [gpu_mem_get(id) for id in range(pynvml.nvmlDeviceGetCount())]

def gpu_with_max_free_mem():
    "get [gpu_id, its_free_ram] for gpu with highest available RAM"
    # returns (None, 0) if no gpu is available
    mem = np.array(gpu_mem_get_all())
    if not len(mem): return (None, 0)
    id = np.argmax(mem[:,Memory.FREE])
    return (id, mem[id,Memory.FREE])
