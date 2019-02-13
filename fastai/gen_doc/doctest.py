import sys, inspect, re
from os.path import basename, split

__all__ = ['this_tests']

def this_tests(testedapi):
     RegisterTestsperAPI.this_tests(testedapi)

def full_name_with_qualname(klass):
     return f'{klass.__module__}.{klass.__qualname__}' 

def set_default(obj):
     if isinstance(obj, set):
          return list(obj)
     raise TypeError 

def get_parent_func(lineno, lines):
    for idx,l in enumerate(reversed(lines[:lineno])):
        if re.match(f'^def test', l):
            return (lineno - (idx+1)), l
    return None

def get_lines(file):
    with open(file, 'r') as f:
          return f.readlines()

class RegisterTestsperAPI:
    apiTestsMap = dict()
    @staticmethod
    def this_tests(testedapi):
        previous_frame = inspect.currentframe().f_back.f_back 
        (pathfilename, line_number, test_function_name, lines, index) = inspect.getframeinfo(previous_frame)
        lineno_parentfunc, parent_func = get_parent_func(line_number, get_lines(pathfilename)) 
        list_test = [{'file': basename(pathfilename), 'test': test_function_name , 'line': str(lineno_parentfunc)}]
        fq_apiname = full_name_with_qualname(testedapi)
        if(fq_apiname in RegisterTestsperAPI.apiTestsMap):
            RegisterTestsperAPI.apiTestsMap[fq_apiname] = RegisterTestsperAPI.apiTestsMap[fq_apiname]  + list_test
        else:
            RegisterTestsperAPI.apiTestsMap[fq_apiname] =  list_test   
      


