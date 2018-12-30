# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import pytest, sys, re
from os.path import abspath, dirname

# make sure we test against the checked out git version of fastai and
# not the pre-installed version. With 'pip install -e .[dev]' it's not
# needed, but it's better to be safe and ensure the git path comes
# second in sys.path (the first path is the test dir path)
git_repo_path = abspath(dirname(dirname(__file__)))
sys.path.insert(1, git_repo_path)

def pytest_addoption(parser):
    parser.addoption( "--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption( "--skipint", action="store_true", default=False, help="skip integration tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipint"):
        skip_int = pytest.mark.skip(reason="--skipint used to skip integration test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_int)

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords: item.add_marker(skip_slow)
