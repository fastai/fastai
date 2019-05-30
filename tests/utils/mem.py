""" Helper functions for dealing with memory usage testing """

import pytest, torch, time
from fastai.utils.mem import *
from math import isclose

# torch.cuda.is_available() checks if we can use NVIDIA GPU. It automatically
# handles the case when CUDA_VISIBLE_DEVICES="" env var is set, so even if CUDA
# is available it will return False, thus we can emulate non-CUDA environment.
use_gpu = torch.cuda.is_available()

# This must run before any tests that measure gpu RAM
# force pytorch to load cuDNN and its kernels to claim unreclaimable memory (~0.5GB) if it hasn't done so already, so that we get correct measurements
def torch_preload_mem():
    if use_gpu: torch.ones((1, 1)).cuda()

def gpu_mem_consume_some(n): return torch.ones((n, n)).cuda()
def gpu_mem_consume_16mb(): return gpu_mem_consume_some(2000)
def gpu_cache_clear(): torch.cuda.empty_cache()
def gpu_mem_reclaim(): gc.collect(); gpu_cache_clear()

def gpu_mem_allocate_mbs(n):
    "Allocate `n` MBs, return the var holding it on success, None on failure. Granularity is of 2MB (mem page size)"
    try:    return torch.ones((n*2**18)).cuda().contiguous()
    except: return None

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

# must cleanup after some previously run tests that may leaked memory,
# before starting this sensitive measurement-wise test
def gpu_prepare_clean_slate(): gc.collect()

# ensure a thread gets a chance to run by creating a tiny pause
def yield_to_thread(): time.sleep(0.001)

########################## validation helpers ###############################
# these functions are for checking expected vs received (actual) memory usage in tests

# number strings can be 1,000.05, so strip commas and convert to float
def str2flt(s): return float(s.replace(',',''))

# mtrace is GPUMemTrace.data output
# ctx is useful as a hint for telling where in the test the trace was measured
def check_mtrace(used_exp, peaked_exp, mtrace, abs_tol=2, ctx=None):
    used_rcv, peaked_rcv = mtrace.data()
    check_mem(used_exp, peaked_exp, used_rcv, peaked_rcv, abs_tol=abs_tol, ctx=ctx)

def check_mem(used_exp, peaked_exp, used_rcv, peaked_rcv, abs_tol=2, ctx=None):
    ctx = f" ({ctx})" if ctx is not None else ""
    assert isclose(used_exp,   used_rcv,   abs_tol=abs_tol), f"used mem: expected={used_exp} received={used_rcv}{ctx}"
    assert isclose(peaked_exp, peaked_rcv, abs_tol=abs_tol), f"peaked mem: expected={peaked_exp} received={peaked_rcv}{ctx}"

# instead of asserting the following print outs the actual mem usage, to aid in debug
# in order not to the change tests simply override `check_mem` with `report_mem` below
def report_mem(used_exp, peaked_exp, used_rcv, peaked_rcv, abs_tol=2, ctx=None):
    ctx = f" ({ctx})" if ctx is not None else ""
    print(f"got:△used={used_rcv}MBs, △peaked={peaked_rcv}MBs{ctx}")

# parses mtrace repr and also asserts that the `ctx` is in the repr
def parse_mtrace_repr(mtrace_repr, ctx):
    "parse the `mtrace` repr and return `used`, `peaked` ints"
    # extract numbers + check ctx matches
    match = re.findall(fr'△Used Peaked MB: +([\d,]+) +([\d,]+) +\({ctx}\)', mtrace_repr)
    assert match, f"input: cs.out={mtrace_repr}, ctx={ctx}"
    used, peaked = map(str2flt, match[0])
    return used, peaked


#check_mem = report_mem
