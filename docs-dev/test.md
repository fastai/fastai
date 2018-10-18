# Testing fastai

<!--ts-->

Table of Contents
-----------------

   * [Testing fastai](#testing-fastai)
      * [Table of Contents](#table-of-contents)
      * [Notebook integration tests](#notebook-integration-tests)
      * [Automated tests](#automated-tests)
         * [Choosing which tests to run](#choosing-which-tests-to-run)
         * [To GPU or not to GPU](#to-gpu-or-not-to-gpu)
         * [Report each sub-test name and its progress](#report-each-sub-test-name-and-its-progress)
         * [Output capture](#output-capture)
         * [Color control](#color-control)
         * [Sending test report to online pastebin service](#sending-test-report-to-online-pastebin-service)
   * [Writing Tests](#writing-tests)
      * [Hints](#hints)
         * [Skipping tests](#skipping-tests)
         * [Getting reproducible results](#getting-reproducible-results)
<!--te-->

## Notebook integration tests

Currently we have few automated tests, so most testing is through integration tests done in Jupyter Notebooks. The two places you should check for notebooks to test your code with are:

 - [The fastai examples](https://github.com/fastai/fastai/tree/master/examples)
 - [The fastai_docs notebooks](https://github.com/fastai/fastai_docs/tree/master/docs_src)

In each case, look for notebooks that have names starting with the application you're working on - e.g. 'text' or 'vision'.


## Automated tests

At the moment there are only a few automated tests, so we need to start expanding it! It's not easy to properly automatically test ML code, but there's lots of opportunities for unit tests.

We use [pytest](https://docs.pytest.org/en/latest/). Here is a complete [pytest API reference](https://docs.pytest.org/en/latest/reference.html).

The tests have been configured to automatically run against the git checked out `fastai` repository and not pre-installed `fastai`.

### Choosing which tests to run

To run all the tests:


   ```
   make test
   ```

or


   ```
   python setup.py test
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

### Slow integration tests

Use it like this (e.g):

   ```
   @pytest.mark.slow
   class TestVisionEndToEnd():
   ```

Tests marked in that way won’t be run on `make test`. To run them, simply say:

   ```
   pytest --runslow
   ```

This marker is designed for end to end training integration tests that probably want a GPU, may need internet access (at least the first time) to download data, and may need manual inspection of failures.




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

## Hints


### Skipping tests

This is useful when a bug is found and a new test is written, yet the bug is not fixed yet. In order to be able to commit it to the main repository we need make sure it's skipped during `make test`.

Methods:

* A **skip** means that you expect your test to pass only if some conditions are met, otherwise pytest should skip running the test altogether. Common examples are skipping windows-only tests on non-windows platforms, or skipping tests that depend on an external resource which is not available at the moment (for example a database).

* A **xfail** means that you expect a test to fail for some reason. A common example is a test for a feature not yet implemented, or a bug not yet fixed. When a test passes despite being expected to fail (marked with pytest.mark.xfail), it’s an xpass and will be reported in the test summary.

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

In order for tests to be reliable the test result should not be random (most of the time).

To get identical reproducable results set, depending on whether you are using `torch`'s random functions, or python's (`numpy`) or both:

* torch RNG

   ```
   import torch
   torch.manual_seed(42)
   ```

* python RNG

   ```
   random.seed(42)
   ```
