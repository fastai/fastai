---
title: Testing fastai
---

## Automated tests

At the moment there are only a few automated tests, so we need to start expanding it! It's not easy to properly automatically test ML code, but there's lots of opportunities for unit tests.

We use [pytest](https://docs.pytest.org/en/latest/). Here is a complete [pytest API reference](https://docs.pytest.org/en/latest/reference.html).

The tests have been configured to automatically run against the git checked out `fastai` repository and not pre-installed `fastai`.

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

### Writing tests

When writing tests:

- Avoid mocks; instead, think about how to create a test of the real functionality that runs quickly
- Use module scope fixtures to run init code that can be shared amongst tests
- Avoid pretrained models, since they have to be downloaded from the internet to run the test
- Create some minimal data for your test, or use data already in repo's data/ directory

### Clearing state

CI builds and when isolation is important (against speed), cache should be cleared:

   ```
   pytest --cache-clear tests
   ```



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

To disable capturing and get the output normally use `-s` or `--capture=no`:

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



# Writing Tests

XXX: Needs to be written. Contributions are welcome.

Until then look at the existing tests.


## Coverage

When you run:

   ```
   make coverage
   ```

it will run the test suite directly via `pytest` and on completion open a browser to show you the coverage report, which will give you an indication of which parts of the code base haven't been exercised by tests yet. So if you are not sure which new tests to write this output can be of great insight.

Remember, that coverage only indicated which parts of the code tests have exercised. It can't tell anything about the quality of the tests. As such, you may have a 100% coverage and a very poorly performing code.


## Hints


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

### fastai/examples/*ipynb

You can run each of these interactively in jupyter, or as CLI:

```
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to notebook examples/tabular.ipynb
```

This set is examples and there is no pass/fail other than visual observation.

