import pytest, fastai
from fastai.utils.mem import *
from utils.mem import *
from math import isclose
# Important: When modifying this test module, make sure to validate that it runs w/o
# GPU, by running: CUDA_VISIBLE_DEVICES="" pytest

# most tests are run regardless of cuda available or not, we just get zeros when gpu is not available
use_gpu = torch.cuda.is_available()
torch_preload_mem()

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
    total, used, free = gpu_mem_get()
    if use_gpu: check_gpu_mem_non_zeros(total, used, free)
    else: check_gpu_mem_zeros(total, used, free)

    # wrong id that can't exist
    check_gpu_mem_zeros(*gpu_mem_get(99))

def test_gpu_mem_all():
    # all available gpus
    mem_per_id = gpu_mem_get_all()
    if use_gpu:
        for mem in mem_per_id: check_gpu_mem_non_zeros(*mem)
    else:
        assert len(mem_per_id) == 0

def test_gpu_with_max_free_mem():
    # all available gpus
    id, free = gpu_with_max_free_mem()
    if use_gpu:
        assert id != None, "have gpu id"
        assert free > 0,   "have gpu free ram"
    else:
        assert id == None, "have no gpu id"
        assert free == 0,  "have no gpu free ram"

@pytest.mark.skipif(not use_gpu, reason="requires cuda")
def test_gpu_mem_measure_consumed_reclaimed():
    gpu_mem_reclaim()
    used_before = gpu_mem_get_used()

    # 1. measure memory consumption
    x1 = gpu_mem_consume_16mb();
    used_after = gpu_mem_get_used()
    diff_real = used_after - used_before
    diff_expected_min = 15 # could be slightly different
    assert diff_real >= diff_expected_min, f"check gpu consumption, expected at least {diff_expected_min}, got {diff_real} diff"

    # 2. measure memory reclamation
    del x1 # this may or may not trigger automatic gc.collect - can't rely on that
    gpu_mem_reclaim() # force gc.collect and cache clearing
    used_after_reclaimed = gpu_mem_get_used()
    # allow 2mb tolerance for rounding of 1 mb on each side
    assert isclose(used_before, used_after_reclaimed, abs_tol=2), f"reclaim all consumed memory, started with {used_before}, now {used_after_reclaimed} used"
