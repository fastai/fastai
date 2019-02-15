# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import pytest, sys, re, json
from os.path import abspath, dirname, join
from utils.mem import use_gpu
from fastai.gen_doc.doctest import RegisterTestsPerAPI, DB_NAME

# make sure we test against the checked out git version of fastai and
# not the pre-installed version. With 'pip install -e .[dev]' it's not
# needed, but it's better to be safe and ensure the git path comes
# second in sys.path (the first path is the test dir path)
git_repo_path = abspath(dirname(dirname(__file__)))
sys.path.insert(1, git_repo_path)

def pytest_addoption(parser):
    parser.addoption( "--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption( "--skipint", action="store_true", default=False, help="skip integration tests")

def mark_items_with_keyword(items, marker, keyword):
    for item in items:
        if keyword in item.keywords: item.add_marker(marker)

def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipint"):
        skip_int = pytest.mark.skip(reason="--skipint used to skip integration test")
        mark_items_with_keyword(items, skip_int, "integration")

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        mark_items_with_keyword(items, skip_slow, "slow")

    if not use_gpu:
        skip_cuda = pytest.mark.skip(reason="CUDA is not available")
        mark_items_with_keyword(items, skip_cuda, "cuda")

@pytest.fixture(scope="session", autouse=True)
def start_doctest_collector(request):
    matching = [s for s in set(sys.argv) if re.match(r'.*test_\w+\.py',s)]
    if not matching: request.addfinalizer(stop_doctest_collector)

def set_default(obj):
     if isinstance(obj, set): return list(obj)
     raise TypeError

def stop_doctest_collector():
    fastai_dir = abspath(join(dirname( __file__ ), '..', 'fastai'))
    with open(fastai_dir + f'/{DB_NAME}', 'w') as f:
        json.dump(obj=RegisterTestsPerAPI.apiTestsMap, fp=f, indent=4, sort_keys=True, default=set_default)
