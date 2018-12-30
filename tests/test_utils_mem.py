import pytest, fastai
from fastai.utils.mem import *
from math import isclose
# Important: When modifying this test module, make sure to validate that it runs w/o
# GPU, by running: CUDA_VISIBLE_DEVICES="" pytest

# most tests are run regardless of cuda available or not, we just get zeros when gpu is not available
if torch.cuda.is_available():
    have_cuda = 1
if "CUDA_VISIBLE_DEVICES" in os.environ and not len(os.environ["CUDA_VISIBLE_DEVICES"]):
    print('detected no gpu env emulation with CUDA_VISIBLE_DEVICES=""')
    have_cuda = 0

# This must run before any tests:
# force pytorch to load cuDNN and its kernels to claim unreclaimable memory (~0.5GB) if it hasn't done so already, so that we get correct measurements
if have_cuda: torch.ones((1, 1)).cuda()

def gpu_mem_consume_some(n): return torch.ones((n, n)).cuda()
def gpu_mem_consume_16mb(): return gpu_mem_consume_some(2000)
def gpu_cache_clear(): torch.cuda.empty_cache()
def gpu_mem_reclaim(): gc.collect(); gpu_cache_clear()

def check_gpu_mem_zeros(total, used, free):
    assert total == 0, "have total GPU RAM"
    assert used  == 0, "have used GPU RAM"
    assert free  == 0, "have free GPU RAM"

def check_gpu_mem_non_zeros(total, used, free):
    assert total > 0, "have total GPU RAM"
    assert used  > 0, "have used GPU RAM"
    assert free  > 0, "have free GPU RAM"

def test_gpu_mem_by_id():
    # test by currently selected device
    total, used, free = get_gpu_mem()
    if have_cuda: check_gpu_mem_non_zeros(total, used, free)
    else: check_gpu_mem_zeros(total, used, free)

    # wrong id that can't exist
    check_gpu_mem_zeros(*get_gpu_mem(99))

def test_gpu_mem_all():
    # all available gpus
    mem_per_id = get_gpu_mem_all()
    if have_cuda:
        for mem in mem_per_id: check_gpu_mem_non_zeros(*mem)
    else:
        assert len(mem_per_id) == 0

def test_gpu_with_max_free_mem():
    # all available gpus
    id, free = get_gpu_with_max_free_mem()
    if have_cuda:
        assert id != None, "have gpu id"
        assert free > 0,   "have gpu free ram"
    else:
        assert id == None, "have no gpu id"
        assert free == 0,  "have no gpu free ram"

@pytest.mark.skipif(not have_cuda, reason="requires cuda")
def test_gpu_mem_measure_consumed_reclaimed():
    gpu_mem_reclaim()
    used_before = get_gpu_mem()[1]

    # 1. measure memory consumption
    x1 = gpu_mem_consume_16mb();
    used_after = get_gpu_mem()[1]
    diff_real = used_after - used_before
    diff_expected_min = 15 # could be slightly different
    assert diff_real >= diff_expected_min, f"check gpu consumption, expected at least {diff_expected_min}, got {diff_real} diff"

    # 2. measure memory reclamation
    del x1 # this may or may not trigger automatic gc.collect - can't rely on that
    gpu_mem_reclaim() # force gc.collect and cache clearing
    used_after_reclaimed = get_gpu_mem()[1]
    # allow 2mb tolerance for rounding of 1 mb on each side
    assert isclose(used_before, used_after_reclaimed, abs_tol=2), f"reclaim all consumed memory, started with {used_before}, now {used_after_reclaimed} used"
