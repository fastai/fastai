import inspect
import json

# customise Makefile
# add cmd param to allow saving
# why desctructor fo saving on --> function alone just does never a safe
# maybe use an object really why is this set default method required
# should this be below docs really?


def full_name_with_qualname(klass):
     return f'{klass.__module__}.{klass.__qualname__}' ##  __name__ for of qualname for short version

def set_default(obj):
     if isinstance(obj, set):
          return list(obj)
     raise TypeError 

## TO DO: find path variable somewhere

def loadTestAPIRegister():
     with open('./fastai/TestAPIRegister.json', 'r') as f:
          json.load(f)           

def saveTestAPIRegister():
     print('\n\n @@@ RegisterTestsperAPI.apiTestsMap \n\n')
     print(RegisterTestsperAPI.apiTestsMap)
     encoded_map = json.dumps(RegisterTestsperAPI.apiTestsMap, indent=2, default=set_default)
     print('\n\n @@@ encoded_map \n\n')
     print(encoded_map)
     with open('./fastai/TestAPIRegister.json', 'w') as f:
          json.dump(encoded_map,f, indent = 2)           

def this_tests(testedapi):
     previous_frame = inspect.currentframe().f_back 
     (filename, line_number, 
     test_function_name, lines, index) = inspect.getframeinfo(previous_frame)
     
     #print('\n\t filename: ' + str(filename))
     #print('\n\t test_function_name: ' + str(test_function_name))
     list_test = [{'file: '+ filename, 'test: ' + test_function_name}]
     #print(list_test)  
     fq_apiname = full_name_with_qualname(testedapi)
     #print('\n\t' + fq_apiname)
     if(fq_apiname in RegisterTestsperAPI.apiTestsMap):
          RegisterTestsperAPI.apiTestsMap[fq_apiname] = RegisterTestsperAPI.apiTestsMap[fq_apiname]  + list_test
     else:
          RegisterTestsperAPI.apiTestsMap[fq_apiname] =  list_test   

class RegisterTestsperAPI():
     apiTestsMap = dict()


   
   
     
        