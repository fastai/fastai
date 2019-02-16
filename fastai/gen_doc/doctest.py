import sys, inspect, re
from os.path import basename, split
from pathlib import Path

__all__ = ['this_tests']

DB_NAME = 'test_api_db.json'

class RegisterTestsPerAPI:
    apiTestsMap = dict()

    @staticmethod
    def this_tests(*testedapis):
        prev_frame = inspect.currentframe().f_back.f_back
        pathfilename, line_number, test_function_name, lines, index = inspect.getframeinfo(prev_frame)
        lineno_parentfunc, parent_func = get_parent_func(line_number, get_lines(pathfilename))
        list_test = [{'file': relative_test_path(pathfilename), 'test': test_function_name , 'line': lineno_parentfunc}]
        for api in testedapis:
             fq_apiname = full_name_with_qualname(api)
             if fq_apiname in RegisterTestsPerAPI.apiTestsMap:
                 RegisterTestsPerAPI.apiTestsMap[fq_apiname] += list_test
             else:
                 RegisterTestsPerAPI.apiTestsMap[fq_apiname] = list_test

def this_tests(*testedapis): RegisterTestsPerAPI.this_tests(*testedapis)

def full_name_with_qualname(testedapi):
    if inspect.ismodule(testedapi): return testedapi.__name__
    name = testedapi.__qualname__ if hasattr(testedapi, '__qualname__') else testedapi.__name__
    return f'{testedapi.__module__}.{name}'

def set_default(obj):
    if isinstance(obj, set): return list(obj)
    raise TypeError

def get_parent_func(lineno, lines, ignore_missing=False):
    "Find any lines where `elt` is called and return the parent test function"
    for idx,l in enumerate(reversed(lines[:lineno])):
        if re.match(f'\s*def test', l):  return (lineno - (idx+1)), l
        if re.match(f'^\w+', l):  break # Top level indent - break because we are out of function scope
    if ignore_missing: return None
    raise LookupError('Could not find parent function for line:', lineno, lines[:lineno])

def relative_test_path(test_file:Path)->str:
    "Path relative to 'fastai' parent directory"
    test_file = Path(test_file)
    testdir_idx = list(reversed(test_file.parts)).index('tests')
    return '/'.join(test_file.parts[-(testdir_idx+1):])

def get_lines(file):
    with open(file, 'r') as f: return f.readlines()
