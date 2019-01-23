""" Helper functions for dealing with memory usage testing """

import pytest, fastai, torch
from fastai.utils.mem import *

# check if we can/should use nvidia gpu
# handle the case where cuda is available, but it is pretended not to have one through setting env var CUDA_VISIBLE_DEVICES=""
def can_use_gpu():
    use_gpu = 1 if torch.cuda.is_available() else 0
    if "CUDA_VISIBLE_DEVICES" in os.environ and not len(os.environ["CUDA_VISIBLE_DEVICES"]):
        # print('detected no gpu env emulation with CUDA_VISIBLE_DEVICES=""')
        use_gpu = 0
    return use_gpu
use_gpu = can_use_gpu()

# This must run before any tests that measure gpu RAM
# force pytorch to load cuDNN and its kernels to claim unreclaimable memory (~0.5GB) if it hasn't done so already, so that we get correct measurements
def torch_preload_mem():
    if use_gpu: torch.ones((1, 1)).cuda()

def gpu_mem_consume_some(n): return torch.ones((n, n)).cuda()
def gpu_mem_consume_16mb(): return gpu_mem_consume_some(2000)
def gpu_cache_clear(): torch.cuda.empty_cache()
def gpu_mem_reclaim(): gc.collect(); gpu_cache_clear()
def gpu_mem_get_used(): return gpu_mem_get().used
def gpu_mem_get_free(): return gpu_mem_get().free

def gpu_mem_allocate_mbs(n):
    " allocate n MBs, return the var holding it on success, None on failure "
    try:
        d = int(2**9*n**0.5)
        return torch.ones((d, d)).cuda().contiguous()
    except:
        return None

# this is very useful if the test needs to hit OOM, so this function will leave
# just the requested amount of GPU free, regardless of GPU utilization or size
# of the card
def gpu_mem_leave_free_mbs(n):
    " consume whatever memory is needed so that n MBs are left free "
    avail = gpu_mem_get_free()
    assert avail > n, f"already have less available mem than desired {n}MBs"
    consume = avail - n
    #print(f"consuming {consume}MB to bring free mem to {n}MBs")
    return gpu_mem_allocate_mbs(consume, fatal=True)
