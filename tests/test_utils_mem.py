import pytest
from fastai.utils.mem import *
from fastai.gen_doc.doctest import this_tests
from utils.mem import *
from utils.text import *
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
    this_tests(gpu_mem_get)
    # test by currently selected device
    total, used, free = gpu_mem_get()
    if use_gpu: check_gpu_mem_non_zeros(total, used, free)
    else: check_gpu_mem_zeros(total, used, free)

    # wrong id that can't exist
    check_gpu_mem_zeros(*gpu_mem_get(99))

def test_gpu_mem_all():
    # all available gpus
    this_tests(gpu_mem_get_all)
    mem_per_id = gpu_mem_get_all()
    if use_gpu:
        for mem in mem_per_id: check_gpu_mem_non_zeros(*mem)
    else:
        assert len(mem_per_id) == 0

def test_gpu_with_max_free_mem():
    this_tests(gpu_with_max_free_mem)
    # all available gpus
    id, free = gpu_with_max_free_mem()
    if use_gpu:
        assert id != None, "have gpu id"
        assert free > 0,   "have gpu free ram"
    else:
        assert id == None, "have no gpu id"
        assert free == 0,  "have no gpu free ram"

@pytest.mark.cuda
def test_gpu_mem_measure_consumed_reclaimed():
    this_tests(gpu_mem_get_used)
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

@pytest.mark.cuda
def test_gpu_mem_trace():

    gpu_prepare_clean_slate()

    mtrace = GPUMemTrace()
    this_tests(mtrace.__class__)

    ### 1. more allocated, less released, then all released, w/o counter reset
    # expecting used=~10, peaked=~15
    x1 = gpu_mem_allocate_mbs(10)
    x2 = gpu_mem_allocate_mbs(15)
    del x2
    yield_to_thread() # hack: ensure peak thread gets a chance to measure the peak
    check_mtrace(used_exp=10, peaked_exp=15, mtrace=mtrace, abs_tol=2, ctx="rel some")

    # check `report`'s format including the right numbers
    ctx = "whoah"
    with CaptureStdout() as cs: mtrace.report(ctx)
    used, peaked = parse_mtrace_repr(cs.out, ctx)
    check_mem(used_exp=10,   peaked_exp=15,
              used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="trace `report`")

    # release the remaining allocation, keeping the global counter running w/o reset
    # expecting used=~0, peaked=~25
    del x1
    check_mtrace(used_exp=0, peaked_exp=25, mtrace=mtrace, abs_tol=2, ctx="rel all")

    ### 2. more allocated, less released, then all released, w/ counter reset
    # expecting used=~10, peaked=~15
    x1 = gpu_mem_allocate_mbs(10)
    x2 = gpu_mem_allocate_mbs(15)
    yield_to_thread() # hack: ensure peak thread gets a chance to measure the peak
    del x2
    check_mtrace(used_exp=10, peaked_exp=15, mtrace=mtrace, abs_tol=2, ctx="rel some")

    # release the remaining allocation, resetting the global counter
    mtrace.reset()
    # expecting used=-10, peaked=0
    del x1
    check_mtrace(used_exp=-10, peaked_exp=0, mtrace=mtrace, abs_tol=2, ctx="rel all")

    # test context + subcontext
    ctx = 'test2'
    mtrace = GPUMemTrace(ctx=ctx)
    mtrace.start() # not needed, calling for testing
    check_mtrace(used_exp=0, peaked_exp=0, mtrace=mtrace, abs_tol=2, ctx=ctx)
    # 1. main context
    with CaptureStdout() as cs: mtrace.report()
    used, peaked = parse_mtrace_repr(cs.out, ctx)
    check_mem(used_exp=0,    peaked_exp=0,
              used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="auto-report on exit")
    # 2. context+sub-context
    subctx = 'sub-context test'
    with CaptureStdout() as cs: mtrace.report(subctx)
    used, peaked = parse_mtrace_repr(cs.out, f'{ctx}: {subctx}')
    check_mem(used_exp=0,    peaked_exp=0,
              used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="auto-report on exit")

    mtrace.stop()

@pytest.mark.cuda
def test_gpu_mem_trace_ctx():
    # context manager
    # expecting used=20, peaked=0, auto-printout
    with CaptureStdout() as cs:
        with GPUMemTrace() as mtrace:
            x1 = gpu_mem_allocate_mbs(20)
    _, _ = parse_mtrace_repr(cs.out, "exit")
    this_tests(mtrace.__class__)
    check_mtrace(used_exp=20, peaked_exp=0, mtrace=mtrace, abs_tol=2, ctx="ctx manager")
    del x1

    # auto-report on exit w/ context and w/o
    for ctx in [None, "test"]:
        with CaptureStdout() as cs:
            with GPUMemTrace(ctx=ctx):
                # expecting used=20, peaked=0
                x1 = gpu_mem_allocate_mbs(20)
        if ctx is None: ctx = "exit" # exit is the hardcoded subctx for ctx manager
        else:           ctx += ": exit"
        used, peaked = parse_mtrace_repr(cs.out, ctx)
        check_mem(used_exp=20,   peaked_exp=0,
                  used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="auto-report on exit")
        del x1

    # auto-report off
    ctx = "auto-report off"
    with CaptureStdout() as cs:
        with GPUMemTrace(ctx=ctx, on_exit_report=False): 1
    assert len(cs.out) == 0, f"stdout: {cs.out}"


# setup for test_gpu_mem_trace_decorator
@gpu_mem_trace
def experiment1(): pass

class NewTestExp():
    @staticmethod
    @gpu_mem_trace
    def experiment2(): pass

@pytest.mark.cuda
def test_gpu_mem_trace_decorator():
    this_tests(gpu_mem_trace)

    # func
    with CaptureStdout() as cs: experiment1()
    used, peaked = parse_mtrace_repr(cs.out, "experiment1: exit")
    check_mem(used_exp=0,    peaked_exp=0,
              used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="")

    # class func
    with CaptureStdout() as cs: NewTestExp.experiment2()
    used, peaked = parse_mtrace_repr(cs.out, "NewTestExp.experiment2: exit")
    check_mem(used_exp=0,    peaked_exp=0,
              used_rcv=used, peaked_rcv=peaked, abs_tol=2, ctx="")


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    #Removed from debugging
    columns = df.columns
    #.drop('index')

    for col in columns:
        col_type = df[col].dtype
        if str(col_type) != 'category' and col_type != 'datetime64[ns]' and col_type != bool:
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        #df[col] = df[col].astype(np.float16)
                    #Sometimes causes and error and had to remove
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        print('Error '+col+' value would be a float64. Disregarding.')
            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
