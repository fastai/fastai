import pytest
import re

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

