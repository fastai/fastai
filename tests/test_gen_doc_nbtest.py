import pytest
from fastai.gen_doc.nbtest import *
from fastai.gen_doc import nbtest
from fastai.gen_doc.doctest import this_tests
import inspect

def test_submodule_name():
    result:str = nbtest._submodule_name(nbtest.doctest)
    assert result == 'gen_doc', 'should return submodule'

    from fastai.core import ifnone
    result:str = nbtest._submodule_name(ifnone)
    assert result == None, f'fastai/module should not have a submodule: {result}'

def test_is_file_match():
    import fastai.text.data
    result = nbtest._is_file_match(fastai.text.data, 'test_text_data.py')
    assert result is not None, f"matches test files with submodule"

    import fastai.core
    result = nbtest._is_file_match(fastai.core.ifnone, 'test_core_subset_category.py')
    assert result is not None, f"matches module subsets"

def test_direct_test_match():
    fn_name = '_is_file_match'
    lines = inspect.getsourcelines(test_is_file_match)[0]
    result = direct_test_match(fn_name, lines, '')
    assert result[0]['line'] == 0
    assert result[0]['test'] == 'test_is_file_match', "should match private functions"

def test_direct_test_match_class_methods():
    fn_name = 'DataBunch.get'
    lines = ['def test_Databunch_get():', '\tpass', 'def test_Databunch_get_specific_function():']
    result = direct_test_match(fn_name, lines, '')
    assert len(result) == 2

def test_fuzzy_test_match():
    lines = ['def test_mock_function():',
             '    x = mock_function(testedapi)',
             'def test_related():',
             '    return related().mock_function()'
            'def test_substring():',
             '    a.mock_func()._mock_function(1,2)']
    result = fuzzy_test_match('mock_function', lines, None)
    assert result[0]['test'] == 'test_mock_function', 'matches simple function calls'
    assert result[1]['test'] == 'test_related', 'matches related calls'
    assert len(result) == 2, 'should not include test_substring'

    lines = ['def test_without_fcall():',
             '    return None',
             'def helper_func():',
             '    x = func(testedapi)',
             '    x = test_function(testedapi)',
             'x = func()']
    result = fuzzy_test_match('func', lines, None)
    assert len(result) == 0, 'should only find parent test functions with `def test_` prefix'

def test_fuzzy_line_match():
    # Testing _fuzzy_test_match private methods
    result = nbtest._fuzzy_line_match('Databunch.get', ['d = DataBunch()', 'item = d.get(5)'])
    assert len(result) == 1, 'finds class methods'

    result = nbtest._fuzzy_line_match('TextList', ['tl = (TextList.from_df()', '', 'LMTextList()'])
    assert len(result) == 1, 'matches classes'

@pytest.mark.skip(reason="local dev only")
def test_get_tests_dir():
    result:Path = nbtest.get_tests_dir(nbtest)
    assert result.parts[-1] == 'tests', f"Failed: get_tests_dir return unexpected result: {result}"

def test_this_tests():

    # function by reference (and self test)
    this_tests(this_tests)

    # multiple entries: same function twice on purpose, should result in just one entry,
    # but also testing multiple entries - and this test tests only a single function.
    this_tests(this_tests, this_tests)

    import fastai
    # explicit fully qualified function (requires all the sub-modules to be loaded)
    this_tests(fastai.gen_doc.doctest.this_tests)

    # explicit fully qualified function as a string
    this_tests('fastai.gen_doc.doctest.this_tests')

    # not a real function
    func = 'foo bar'
    try: this_tests(func)
    except Exception as e: assert f"'{func}' is not a function" in str(e)
    else: assert False, f'this_tests({func}) should have failed'

    # not a function as a string that looks like fastai function, but it is not
    func = 'fastai.gen_doc.doctest.doesntexistreally'
    try: this_tests(func)
    except Exception as e: assert f"'{func}' is not a function" in str(e)
    else: assert False, f'this_tests({func}) should have failed'

    # not a fastai function
    import numpy as np
    func = np.any
    try: this_tests(func)
    except Exception as e: assert f"'{func}' is not in the fastai API" in str(e)
    else: assert False, f'this_tests({func}) should have failed'
