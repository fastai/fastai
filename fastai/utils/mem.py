"Utility functions for memory management"

from ..imports.torch import *
from ..core import *
from ..script import *
from ..utils.env import *
import pynvml, functools, traceback, threading, time
from collections import namedtuple

IS_IN_IPYTHON = is_in_ipython()

GPUMemory = namedtuple('GPUMemory', ['total', 'used', 'free'])

have_cuda = 0
if torch.cuda.is_available():
    pynvml.nvmlInit()
    have_cuda = 1

def preload_pytorch():
    torch.ones((1, 1)).cuda()

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

def gpu_mem_get_used_no_cache():
    torch.cuda.empty_cache()
    return gpu_mem_get().used

def gpu_mem_get_fast_used(gpu_handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return int(info.used/2**20)

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
# ipython has a feature where it stores tb with all the locals() tied in, which
# prevents gc.collect from freeing those variables, therefore we cleanse the tb
# before handing it over to ipython.
#
# under non-ipython environment it doesn't do anything.
#
# under ipython currently it strips tb by default only for the "CUDA out of memory" exception.
#
# The env var FASTAI_TB_CLEAR_FRAMES changes this behavior when run under ipython,
# depending on its value: (os.environ['FASTAI_TB_CLEAR_FRAMES']="0")
#
# "0": never  strip tb (makes it possible to always use %debug magic, but with leaks)
# "1": always strip tb (never need to worry about leaks, but %debug won't work)
#
def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tb_clear_frames = os.environ.get('FASTAI_TB_CLEAR_FRAMES', None)
        if not IS_IN_IPYTHON or tb_clear_frames=="0":
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            if ("CUDA out of memory" in str(e) or
                "device-side assert triggered" in str(e) or
                tb_clear_frames == "1"):
                type, val, tb = get_ref_free_exc_info() # must!
                raise type(val).with_traceback(tb) from None
            else: raise # re-raises the exact last exception
    return wrapper


# if function decorator is not a good option, use context manager, example:
# with gpu_mem_restore_ctx():
#    learn.fit_one_cycle(1,1e-2)
# this particular one will clear tb on any exception
class gpu_mem_restore_ctx():
    "context manager to reclaim RAM if an exception happened under ipython"
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val: return True
        traceback.clear_frames(exc_tb)
        raise exc_type(exc_val).with_traceback(exc_tb) from None


#
# memtrace = GPUMemTrace()
# memtrace.start() # start tracing
#
# some_code()
# memtrace.report() # print intermediary cumulative report
# used, peak =  memtrace.data() # same but as data
#
# some_code()
# memtrace.report('2nd run') # print intermediary cumulative report
# used, peak =  memtrace.data()
#
# for i in range(10):
#     memtrace.reset()
#     code()
#     memtrace.report(f'i={i}') # report for just the last code run since reset
#
# combine report+reset
# memtrace.reset()
# for i in range(10):
#     code()
#     memtrace.report_n_reset(f'i={i}') # report for just the last code run since reset
#
# memtrace.stop() # stop the monitor thread
#

class GPUMemTrace():
    "Trace GPU allocated and peak memory usage"
    def __init__(self, silent=False):
        assert torch.cuda.is_available(), "pytorch CUDA is required"
        self.silent = silent # quickly turn off printouts from the constructor

    def silent(self, silent=False):
        self.silent = silent

    def reset(self):
        self.used_start = gpu_mem_get_used_no_cache()
        self.used_peak  = self.used_start

    def start(self):
        self.reset()
        self.peak_monitor_start()

    def stop(self):
        self.peak_monitor_stop()

    def __del__(self):
        self.stop()

    def data(self):
        self.delta_used = gpu_mem_get_used_no_cache() - self.used_start
        self.delta_peak = self.used_peak              - self.used_start
        return (self.delta_used, self.delta_peak)

    def report_n_reset(self, note=''):
        self.report(note)
        self.reset()

    def report(self, note=''):
        "printout used+delta peak, and an optional context note"
        if self.silent: return
        delta_used, delta_peak = self.data()
        if note: note = f": {note}"
        print(f"△used {delta_used}, △peak {delta_peak}{note}")

    def peak_monitor_start(self):
        self.peak_monitoring = True

        # continually sample RAM usage
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def peak_monitor_stop(self):
        self.peak_monitoring = False

    def peak_monitor_func(self):
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        while True:
            self.used_peak = max(gpu_mem_get_fast_used(gpu_handle), self.used_peak)
            if not self.peak_monitoring: break
            time.sleep(0.001) # 1msec
