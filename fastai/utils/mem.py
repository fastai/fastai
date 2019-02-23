"Utility functions for memory management"

from ..imports.torch import *
from ..core import *
from ..script import *
from ..utils.env import *
import pynvml, functools, traceback, threading, time
from collections import namedtuple
import platform

IS_IN_IPYTHON = is_in_ipython()

use_gpu = torch.cuda.is_available()

GPUMemory = namedtuple('GPUMemory', ['total', 'free', 'used'])

is_osx = platform.system() == "Darwin"

# transparently monkey patch pynvx as pynvml API on OSX (for the few funcs we use)
if use_gpu and is_osx:
    try:
        import pynvx
    except:
        print("please install pynvx on OSX: pip install pynvx")
        sys.exit(1)

    # missing function
    def cudaDeviceGetHandleByIndex(id): return pynvx.cudaDeviceGetHandles()[id]
    setattr(pynvx, 'cudaDeviceGetHandleByIndex', cudaDeviceGetHandleByIndex)

    # different named and return value needs be a named tuple
    def cudaDeviceGetMemoryInfo(handle):
        info = pynvx.cudaGetMemInfo(handle)
        return GPUMemory(*info)
    setattr(pynvx, 'cudaDeviceGetMemoryInfo', cudaDeviceGetMemoryInfo)

    # remap the other functions
    for m in ['Init', 'DeviceGetCount', 'DeviceGetHandleByIndex', 'DeviceGetMemoryInfo']:
        setattr(pynvx, f'nvml{m}', getattr(pynvx, f'cuda{m}'))
    pynvml = pynvx

if use_gpu:
    pynvml.nvmlInit()

def preload_pytorch():
    torch.ones((1, 1)).cuda()

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)

def gpu_mem_get(id=None):
    "get total, used and free memory (in MBs) for gpu `id`. if `id` is not passed, currently selected torch device is used"
    if not use_gpu: return GPUMemory(0, 0, 0)
    if id is None: id = torch.cuda.current_device()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMemory(*(map(b2mb, [info.total, info.free, info.used])))
    except:
        return GPUMemory(0, 0, 0)

def gpu_mem_get_all():
    "get total, used and free memory (in MBs) for each available gpu"
    if not use_gpu: return []
    return list(map(gpu_mem_get, range(pynvml.nvmlDeviceGetCount())))

def gpu_mem_get_free_no_cache():
    "get free memory (in MBs) for the currently selected gpu id, after emptying the cache"
    torch.cuda.empty_cache()
    return gpu_mem_get().free

def gpu_mem_get_used_no_cache():
    "get used memory (in MBs) for the currently selected gpu id, after emptying the cache"
    torch.cuda.empty_cache()
    return gpu_mem_get().used

def gpu_mem_get_used_fast(gpu_handle):
    "get used memory (in MBs) for the currently selected gpu id, w/o emptying the cache, and needing the `gpu_handle` arg"
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return b2mb(info.used)

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
                gc.collect()
                if "device-side assert triggered" in str(e):
                    warn("""When 'device-side assert triggered' error happens, it's not possible to recover and you must restart the kernel to continue. Use os.environ['CUDA_LAUNCH_BLOCKING']="1" before restarting to debug""")
                raise type(val).with_traceback(tb) from None
            else: raise # re-raises the exact last exception
    return wrapper

class gpu_mem_restore_ctx():
    "context manager to reclaim RAM if an exception happened under ipython"
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val: return True
        traceback.clear_frames(exc_tb)
        gc.collect()
        raise exc_type(exc_val).with_traceback(exc_tb) from None

class GPUMemTrace():
    "Trace GPU allocated and peaked memory usage"
    def __init__(self, silent=False):
        assert torch.cuda.is_available(), "pytorch CUDA is required"
        self.silent = silent # quickly turn off printouts from the constructor
        self.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def __repr__(self):
        delta_used, delta_peaked = self.data()
        return f"△used: {delta_used}MB, △peaked: {delta_peaked}MB"

    def silent(self, silent=False):
        self.silent = silent

    def reset(self):
        self.used_start  = gpu_mem_get_used_no_cache()
        self.used_peak   = self.used_start
        self.data_is_set = False

    def start(self):
        self.reset()
        self.peak_monitor_start()

    def stop(self):
        self.data_set()
        self.peak_monitor_stop()

    def __del__(self):
        self.stop()

    def data_set(self):
        self.delta_used   = gpu_mem_get_used_no_cache() - self.used_start
        self.delta_peaked = self.used_peak              - self.used_start - self.delta_used
        self.data_is_set = True

    def data(self):
        if not self.data_is_set: self.data_set()
        return (self.delta_used, self.delta_peaked)

    def report_n_reset(self, note=''):
        self.report(note)
        self.reset()

    def report(self, note=''):
        "printout used+peaked, and an optional context note"
        if self.silent: return
        delta_used, delta_peaked = self.data()
        if note: note = f": {note}"
        print(f"{self}{note}")

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
            self.used_peak = max(gpu_mem_get_used_fast(gpu_handle), self.used_peak)
            if not self.peak_monitoring: break
            time.sleep(0.001) # 1msec
