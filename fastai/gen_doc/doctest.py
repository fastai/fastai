import sys, re, json, pprint
from pathlib import Path
from collections import defaultdict
from inspect import currentframe, getframeinfo, ismodule
from warnings import warn

__all__ = ['this_tests']

DB_NAME = 'test_api_db.json'

def _json_set_default(obj):
    if isinstance(obj, set): return list(obj)
    raise TypeError

class TestAPIRegistry:
    "Tests register which API they validate using this class."
    api_tests_map     = defaultdict(list)
    has_this_tests = None
    missing_this_tests = set()

    # logic for checking whether each test calls `this_tests`:
    # 1. `has_this_tests` is set to False during test's 'setup' stage
    # 2. `this_tests` sets this flag to True when it's successfully completes
    # 3. if during the 'teardown' stage `has_this_tests` is still False then we know that tests needs `has_this_tests`

    @staticmethod
    def this_tests(*funcs):
        prev_frame = currentframe().f_back.f_back
        file_name, lineno, test_name, _, _ = getframeinfo(prev_frame)
        parent_func_lineno, _ = get_parent_func(lineno, get_lines(file_name))
        entry = {'file': relative_test_path(file_name), 'test': test_name , 'line': parent_func_lineno}
        for func in funcs:
            try:
                func_fq = get_func_fq_name(func)
            except:
                raise Exception(f"'{func}' is not a function")
            if re.match(r'fastai\.', func_fq):
                if entry not in TestAPIRegistry.api_tests_map[func_fq]:
                    TestAPIRegistry.api_tests_map[func_fq].append(entry)
            else:
                raise Exception(f"'{func}' is not in the fastai API")
        TestAPIRegistry.has_this_tests = True

    def this_tests_flag_reset(file_name, test_name):
        TestAPIRegistry.has_this_tests = False

    def this_tests_flag_check(file_name, test_name):
        if not TestAPIRegistry.has_this_tests:
            TestAPIRegistry.missing_this_tests.add(f"{file_name}::{test_name}")

    def registry_save():
        if TestAPIRegistry.api_tests_map:
            path = Path(__file__).parent.parent.resolve()/DB_NAME
            print(f"\n*** Saving test registry @ {path}")
            with open(path, 'w') as f:
                json.dump(obj=TestAPIRegistry.api_tests_map, fp=f, indent=4, sort_keys=True, default=_json_set_default)

    def missing_this_tests_alert():
        if TestAPIRegistry.missing_this_tests:
            msg = "\n\n\n*** Warning: Please include `this_tests` call in each of the following:\n{}\n\n".format('\n'.join(sorted(TestAPIRegistry.missing_this_tests)))
            # short warn call on purpose, as pytest re-pastes the code and we want it non-noisy
            warn(msg)

def this_tests(*funcs): TestAPIRegistry.this_tests(*funcs)

def str2func(name):
    "Converts 'fastai.foo.bar' into an function 'object' if such exists"
    if isinstance(name, str): subpaths = name.split('.')
    else:                     return None

    module = subpaths.pop(0)
    if module in sys.modules: obj = sys.modules[module]
    else:                     return None

    for subpath in subpaths:
        obj = getattr(obj, subpath, None)
        if obj == None: return None
    return obj

def get_func_fq_name(func):
    if ismodule(func): return func.__name__
    if isinstance(func, str): func = str2func(func)
    name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__
    return f'{func.__module__}.{name}'

def get_parent_func(lineno, lines, ignore_missing=False):
    "Find any lines where `elt` is called and return the parent test function"
    for idx,l in enumerate(reversed(lines[:lineno])):
        if re.match(f'\s*def test', l):  return (lineno - idx), l # 1 based index for github
        if re.match(f'\w+', l):  break # top level indent - out of function scope
    if ignore_missing: return None
    raise LookupError('Could not find parent function for line:', lineno, lines[:lineno])

def relative_test_path(test_file:Path)->str:
    "Path relative to the `fastai` parent directory"
    test_file = Path(test_file)
    testdir_idx = list(reversed(test_file.parts)).index('tests')
    return '/'.join(test_file.parts[-(testdir_idx+1):])

def get_lines(file):
    with open(file, 'r') as f: return f.readlines()
