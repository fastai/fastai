---
title: Testing fastai
---

## Automated tests

At the moment there are only a few automated tests, so we need to start expanding it! It's not easy to properly automatically test ML code, but there's lots of opportunities for unit tests.

We use [pytest](https://docs.pytest.org/en/latest/). Here is a complete [pytest API reference](https://docs.pytest.org/en/latest/reference.html).

The tests have been configured to automatically run against the `fastai` directory inside the `fastai` git repository and not pre-installed `fastai`. i.e. `tests/test_*` work with `../fastai`.


## Running Tests

### Choosing which tests to run

To run all the tests:

   ```
   pytest
   ```

or:

   ```
   make test
   ```

or:

   ```
   python setup.py test
   ```


To skip the integration tests in order to do quick testing while you work:

   ```
   pytest --skipint
   ```

If you need to skip a certain test module temporarily you can either tell `pytest` which tests to run explicitly, so for example to skip any test modules that contain the string `link`, you could run:

   ```
   pytest `ls -1 tests/*py | grep -v link`
   ```

To run an individual test file:

   ```
   pytest tests/test_core.py
   ```

Run tests by keyword expressions:

   ```
   pytest -k "list and not listify" tests/test_core.py
   ```

For example, if we have the following tests:

   ```
   def test_whatever():
   def test_listify(p, q, expected):
   def test_listy():
   ```

it will first select `test_listify` and `test_listy`, and then deselect `test_listify`, resulting in only the sub-test `test_listy` being run.

More ways: https://docs.pytest.org/en/latest/usage.html

For nuances of configuring pytest's repo-wide behavior see [collection](https://docs.pytest.org/en/latest/example/pythoncollection.html).



### Clearing state

CI builds and when isolation is important (against speed), cache should be cleared:

   ```
   pytest --cache-clear tests
   ```


### Running tests in parallel

This can speed up the total execution time of the test suite.
   ```
   pip install pytest-xdist
   ```
   ```
   $ time pytest
   real    0m51.069s
   $ time pytest -n 6
   real    0m26.940s
   ```
That's twice the speed of the normal sequential execution!

We just need to fix the temp files creation to use a unique string (pid?), otherwise at times some tests collide in a race condition over the same temp file path.


### Test order and repetition

It's good to repeat the tests several times, in sequence, randomly, or in sets, to detect any potential inter-dependency and state-related bugs (tear down). And the straightforward multiple repetition is just good to detect some problems that get uncovered by randomness of DL.

Plugins:

* Repeat tests:

   ```
   pip install pytest-repeat
   ```

   Now 2 new options becomes available:

   ```
  --count=COUNT         Number of times to repeat each test
  --repeat-scope={function,class,module,session} Scope for repeating tests
   ```

   e.g.:
   ```
   pytest --count=10 tests/test_fastai.py
   ```
   ```
   pytest --count=10 --repeat-scope=function tests
   ```



* Run tests in a random order:

   ```
   pip install pytest-random-order
   ```

   Important: Presence of `pytest-random-order` will automatically randomize tests, no configuration change or command line options is required.

   XXX: need to find a package or write our own `pytest` extension to be able to randomize at will, since the two available modules that do that once installed force the randomization by default.

   As explained earlier this allows detection of coupled tests - where one test's state affects the state of another. When `pytest-random-order` is installed it will print the random seed it used for that session, e.g:

   ```
   pytest tests
   [...]
   Using --random-order-bucket=module
   Using --random-order-seed=573663
   [...]
   ```

   So that if the given particular sequence fails, you can reproduce it by adding that exact seed, e.g.:

   ```
   pytest --random-order-seed=573663
   [...]
   Using --random-order-bucket=module
   Using --random-order-seed=573663
   ```

   It will only reproduce the exact order if you use the exact same list of tests (or no list at all (==all)). Once you start to manually narrowing down the list you can no longer rely on the seed, but have to list them manually in the exact order they failed and tell pytest to not randomize them instead using `--random-order-bucket=none`, e.g.:

   ```
   pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
   ```

   To disable the shuffling for all tests:

   ```
   pytest --random-order-bucket=none
   ```

   By default `--random-order-bucket=module` is implied, which will shuffle the files on the module levels. It can also shuffle on `class`, `package`, `global` and `none` levels. For the complete details please see its [documentation](https://github.com/jbasko/pytest-random-order).

Randomization alternatives:

* [`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)

   This module has a very similar functionality/interface, but it doesn't have the bucket modes available in `pytest-random-order`. It has the same problem of imposing itself once installed.


### To GPU or not to GPU


On a GPU-enabled setup, to test in CPU-only mode add `CUDA_VISIBLE_DEVICES=""`:
   ```
   CUDA_VISIBLE_DEVICES="" pytest tests/test_vision.py
   ```

To do the same inside the code of the test:
   ```
   fastai.torch_core.default_device = torch.device('cpu')
   ```

To switch back to cuda:
   ```
   fastai.torch_core.default_device = torch.device('cuda')
   ```

Make sure you don't hard-code any specific device ids in the test, since different users may have a different GPU setup. So avoid code like:
   ```
   fastai.torch_core.default_device = torch.device('cuda:1')
   ```
which tells `torch` to use the 2nd GPU. Instead, if you'd like to run a test locally on a different GPU, use the `CUDA_VISIBLE_DEVICES` environment variable:
   ```
   CUDA_VISIBLE_DEVICES="1" pytest tests/test_vision.py
   ```




### Report each sub-test name and its progress

For a single or a group of tests via `pytest` (after `pip install pytest-pspec`):

   ```
   pytest --pspec tests/test_fastai.py
   pytest --pspec tests
   ```

For all tests via `setup.py`:

   ```
   python setup.py test --addopts="--pspec"
   ```

This also means that meaningful names for each sub-test are important.


### Output capture

During test execution any output sent to `stdout` and `stderr` is captured. If a test or a setup method fails, its according captured output will usually be shown along with the failure traceback.

To disable output capturing and to get the `stdout` and `stderr` normally, use `-s` or `--capture=no`:

   ```
   pytest -s tests/test_core.py
   ```

To send test results to JUnit format output:

   ```
   py.test tests --junitxml=result.xml
   ```


### Color control

To have no color (e.g. yellow on white bg is not readable):

   ```
   pytest --color=no tests/test_core.py
   ```



### Sending test report to online pastebin service

Creating a URL for each test failure:

   ```
   pytest --pastebin=failed tests/test_core.py
   ```

This will submit test run information to a remote Paste service and provide a URL for each failure. You may select tests as usual or add for example -x if you only want to send one particular failure.

Creating a URL for a whole test session log:

   ```
   pytest --pastebin=all tests/test_core.py
   ```




## Writing Tests

When writing tests:

- Avoid mocks; instead, think about how to create a test of the real functionality that runs quickly
- Use module scope fixtures to run init code that can be shared amongst tests
- Avoid pretrained models, since they have to be downloaded from the internet to run the test
- Create some minimal data for your test, or use data already in repo's data/ directory



### Skipping tests

This is useful when a bug is found and a new test is written, yet the bug is not fixed yet. In order to be able to commit it to the main repository we need make sure it's skipped during `make test`.

Methods:

* A **skip** means that you expect your test to pass only if some conditions are met, otherwise pytest should skip running the test altogether. Common examples are skipping windows-only tests on non-windows platforms, or skipping tests that depend on an external resource which is not available at the moment (for example a database).

* A **xfail** means that you expect a test to fail for some reason. A common example is a test for a feature not yet implemented, or a bug not yet fixed. When a test passes despite being expected to fail (marked with pytest.mark.xfail), it’s an xpass and will be reported in the test summary.

One of the important differences between the two is that `skip` doesn't run the test, and `xfail` does. So if the code that's buggy causes some bad state that will affect other tests, do not use `xfail`.

Implementation:

* The whole test unconditionally:

   ```
   @pytest.mark.skip(reason="this bug needs to be fixed")
   def test_feature_x():
   ```

   or the `xfail` way:
   ```
   @pytest.mark.xfail
   def test_feature_x():
   ```



* Based on some internal check inside the test:

   ```
   def test_feature_x():
       if not has_something(): pytest.skip("unsupported configuration")
   ```
   or the whole module:

   ```
   import pytest

   if not pytest.config.getoption("--custom-flag"):
       pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
   ```
   or the `xfail` way:

   ```
   def test_feature_x():
       pytest.xfail("expected to fail until bug XYZ is fixed")
   ```

* Skip all tests in a module if some import is missing:

   ```
   docutils = pytest.importorskip("docutils", minversion="0.3")
   ```

* Skip if

   ```
   import sys
   @pytest.mark.skipif(sys.version_info < (3,6),
                      reason="requires python3.6 or higher")
   def test_feature_x():
   ```
   or the whole module:

   ```
   @pytest.mark.skipif(sys.platform == 'win32',
                      reason="does not run on windows")
   class TestPosixCalls(object):
       def test_feature_x(self):
   ```

More details, example and ways are [here](https://docs.pytest.org/en/latest/skipping.html).



#### Custom markers


Normally, you should be able to declare a test as:

```
import pytest
@pytest.mark.mymarker
def test_mytest(): ...
```

You can then restrict a test run to only run tests marked with `mymarker`:

```
pytest -v -m mymarker
```

Running all tests except the `mymarker` ones:

```
$ pytest -v -m "not mymarker"
```

Custom markers should be registered in `setup.cfg`, for example:

```
[tool:pytest]
# force all used markers to be registered here with an explanation
addopts = --strict
markers =
    marker1: description of its purpose
    marker2: description of its purpose
```

#### fastai custom markers

These are defined in `tests/conftest.py`.

The following markers override normal marker functionality, so they won't work with:

```
pytest -m marker
```

and have their own command line option to be used instead, which are defined in `tests/conftest.py`, and can also be seen in the output of `pytest -h` in the "custom options" section:

```
custom options:
  --runslow             run slow tests
  --skipint             skip integration tests
```

* `slow` - skip tests that can be quite slow (especially on CPU):

   ```
   @pytest.mark.slow
   def test_some_slow_test(): ...
   ```

   To force this kind of tests to run, use:
   ```
   pytest --runslow
   ```

* `integration` - used for tests that are relatively slow but OK to be run on CPU and useful when one needs to finish the tests suite asap (also remember to use parallel testing if that's the case [xdist](#running-tests-in-parallel)). These are usually declared on the module level with:

   ```
   pytestmark = pytest.mark.integration
   ```

   And to skip those use:
   ```
   pytest --skipint
   ```

* `cuda` - mark tests as requiring a CUDA device to run (skipped if no such device is present). These tests check CUDA-specific code, e.g., compiling and running kernels or GPU version of function's `forward`/`backward` methods.


### After test cleanup

To ensure some cleanup code is always run at the end of the test module, add to the desired test module the following code:

```
@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    """Cleanup the tmp file once we are finished."""
    def remove_tmp_file():
        file = "foobar.tmp"
        if os.path.exists(file): os.remove(file)
    request.addfinalizer(remove_tmp_file)
```

The `autouse=True` tells `pytest` to run this fixture automatically (without being called anywhere else).

Use `scope="session"` to run the teardown code not at the end of this test module, but after all test modules were run, i.e. just before `pytest` exits.

Another way to accomplish the global teardown is to put in `tests/conftest.py`:

```
def pytest_sessionfinish(session, exitstatus):
    # global tear down code goes here
```

To run something before and after each test, add to the test module:

```
@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:
    some_setup()
    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:
    some_teardown()
```

`autouse=True` makes this function run for each test defined in the same module automatically.

For creation/teardown of temporary resources for the scope of a test, do the same as above, except get `yield` to return that resource.

```
@pytest.fixture(scope="module")
def learner_obj():
    # Code that will run before your test, for example:
    learn = Learner(...)
    # A test function will be run at this point
    yield learn
    # Code that will run after your test, for example:
    del learn
```

You can now use that function as an argument to a test function:

```
def test_foo(learner_obj):
    learner_obj.fit(...)
```




### Testing the stdout/stderr output

In order to test functions that write to `stdout` and/or `stderr`, the test can access those streams using the `pytest`'s [capsys system](https://docs.pytest.org/en/latest/capture.html). Here is how this is accomplished:

```
import sys
def print_to_stdout(s): print(s)
def print_to_stderr(s): sys.stderr.write(s)
def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr() # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```

And, of course, most of the time, stderr will come as a part of an exception, so try/except has to be used in such a case:

```
def raise_exception(msg): raise ValueError(msg)
def test_something_exception():
    msg = "Not a good value"
    error = ''
    try: raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

Another approach to capturing stdout, is via `contextlib.redirect_stdout`:

```
from io import StringIO
from contextlib import redirect_stdout
def print_to_stdout(s): print(s)
def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer): print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

An important potential issue with capturing stdout is that it may contain `\r` characters that in normal `print` reset everything that has been printed so far. There is no problem with `pytest`, but with `pytest -s` these characters get included in the buffer, so to be able to have the test run w/ and w/o `-s`, you have to make an extra cleanup to the captured output, using `re.sub(r'^.*\r', '', buf, 0, re.M)`. You can use a test helper function for that:

```
from utils.text import *
output = apply_print_resets(output)
```

But, then we have a helper context manager wrapper to automatically take care of it all, regardless of whether it has some `\r`s in it or not, so it's a simple:
```
from utils.text import *
with CaptureStdout() as cs: function_that_writes_to_stdout()
print(cs.out)
```
Here is a full test example:
```
from utils.text import *
msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs: print(msg + final)
assert cs.out == final+"\n", f"captured: {cs.out}, expecting {final}"
```

If you'd like to capture `stderr` use the `CaptureStderr` class instead:

```
from utils.text import *
with CaptureStderr() as cs: function_that_writes_to_stderr()
print(cs.err)
```

If you need to capture both streams at once, use the parent `CaptureStd` class:

```
from utils.text import *
with CaptureStd() as cs: function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```



### Testing memory leaks

This section is currently focused on GPU RAM since it's the scarce resource, but we should test general RAM too.

#### Utils

* Memory measuring helper utils are found in `tests/utils/mem.py`:

   ```
   from utils.mem import *
   ```

* Test whether we can use GPU:
   ```
   use_gpu = torch.cuda.is_available()
   ```
   `torch.cuda.is_available()` checks if we can use NVIDIA GPU. It automatically handles the case when CUDA_VISIBLE_DEVICES="" env var is set, so even if CUDA is available it will return False, thus we can emulate non-CUDA environment.

* Force `pytorch` to preload cuDNN and its kernels to claim unreclaimable memory (~0.5GB) if it hasn't done so already, so that we get correct measurements. This must run before any tests that measure GPU RAM. If you don't run it you will get erratic behavior and wrong measurements.
   ```
   torch_preload_mem()
   ```

* Consume some GPU RAM:
   ```
   gpu_mem_consume_some(n)
   ```
   `n` is the size of the matrix of `torch.ones`. When `n=2**14` it consumes about 1GB, but that's too much for the test suite, so use small numbers, e.g.: `n=2000` consumes about 16MB.


* alias for `torch.cuda.empty_cache()`
   ```
   gpu_cache_clear()
   ```
   It's absolutely essential to run this one, if you're trying to measure real used memory. If cache doesn't get cleared the reported used/free memory can be quite inconsistent.


* This is a combination of `gc.collect()` and `torch.cuda.empty_cache()`
   ```
   gpu_mem_reclaim()
   ```
   Again, this one is crucial for measuring the memory usage correctly. While normal objects get destroyed and their memory becomes available/cached right away, objects with circular references only get freed up when python invokes `gc.collect`, which happens periodically. So if you want to make sure your test doesn't get caught in the inconsistency of getting `gc.collect` to be called during that test or not, call it yourself. But, remember, that if you have to call `gc.collect()` there could be a problem that you will be masking by calling it. So before using it understand what it is doing.

   After `gc.collect()` is called this functions clears the cache that potentially grew due to the released by `gc` objects, and we want to make sure we get the real used/free memory at all times.

* This is a wrapper for getting the used memory for the currently selected device.
   ```
   gpu_mem_get_used()
   ```

#### Concepts

* Taking into account cached memory and unpredictable `gc.collect` calls. See above.

* Memory fluctuations. When measuring either general or GPU RAM there is often a small fluctuation in reported numbers, so when writing tests use functions that approximate equality, but do think deep about the margin you allow, so that the test is useful and yet it doesn't fail at random times.

   Also remember that rounding happens when Bs are converted to MBs.

   Here is an example:
   ```
   from math import isclose
   used_before = gpu_mem_get_used()
   ... some gpu consuming code here ...
   used_after = gpu_mem_get_used()
   assert isclose(used_before, used_after, abs_tol=6), "testing absolute tolerance"
   assert isclose(used_before, used_after, rel_tol=0.02), "testing relative tolerance"
   ```
   This example compares used memory size (in MBs). The first assert compares whether the absolute difference between the two numbers is no more than 6.
   The second assert does the same but uses a relative tolerance in percents -- `0.02` in the example means `2%`. So the accepted difference between the two numbers is no more than `2%`. Often absolute numbers provide a better test, because a percent-based approach could result in quite a large gap if the numbers are big.






### Getting reproducible results

In some situations you may want to remove randomness for your tests. To get identical reproducable results set, you'll need to set `num_workers=1` (or 0) in your DataLoader/DataBunch, and depending on whether you are using `torch`'s random functions, or python's (`numpy`) or both:

* torch RNG

   ```
   import torch
   torch.manual_seed(42)
   torch.backends.cudnn.deterministic = True
   ```

* python RNG

   ```
   random.seed(42)
   ```




### Debugging tests


To start a debugger at the point of the warning, do this:

```
pytest tests/test_vision_data_block.py -W error::UserWarning --pdb
```

### Tests requiring jupyter notebook environment

If [pytest-ipynb](https://github.com/zonca/pytest-ipynb) pytest extension is installed it's possible to add `.ipynb` files to the normal test suite.

Basically, you just write a normal notebook with asserts, and `pytest` just runs it, along with normal `.py` tests, reporting any assert failures normally.

We currently don't have such tests, and if we add any, we will first need to make a conda package for it on the fastai channel, and then add this dependency to fastai.
(note: I haven't researched deeply, perhaps there are other alternatives)

Here is [one example](https://github.com/stas00/ipyexperiments/blob/master/tests/test_cpu.ipynb) of such test.



## Coverage

When you run:

   ```
   make coverage
   ```

it will run the test suite directly via `pytest` and on completion open a browser to show you the coverage report, which will give you an indication of which parts of the code base haven't been exercised by tests yet. So if you are not sure which new tests to write this output can be of great insight.

Remember, that coverage only indicated which parts of the code tests have exercised. It can't tell anything about the quality of the tests. As such, you may have a 100% coverage and a very poorly performing code.


## Notebook integration tests

The two places you should check for notebooks to test your code with are:

 - [The fastai examples](https://github.com/fastai/fastai/tree/master/examples)
 - [The docs_src notebooks](https://github.com/fastai/fastai/tree/master/docs_src)

In each case, look for notebooks that have names starting with the application you're working on - e.g. 'text' or 'vision'.

### docs_src/*ipynb

The `docs_src` notebooks can be executed as a test suite: You need to have at least 8GB available on your GPU to run all of the tests. So make sure you shutdown any unnecessary jupyter kernels, so that the output of your `nvidia-smi` shows that you have at least 8GB free.

```
cd docs_src
./run_tests.sh
```

To run a subset:

```
./run_tests.sh callback*
```

There are a lot more details on this subject matter in this [document](https://github.com/fastai/fastai/blob/master/docs_src/nbval/README.md).

### examples/*ipynb

You can run each of these interactively in jupyter, or as CLI:

```
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to notebook examples/tabular.ipynb
```

This set is examples and there is no pass/fail other than visual observation.
