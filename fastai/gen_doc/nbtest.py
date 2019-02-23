"`gen_doc.nbtest` shows pytest documentation for module functions"

import inspect, os, re
from os.path import abspath, dirname, join
from collections import namedtuple

from fastai.gen_doc import nbdoc
from ..imports.core import *
from .core import ifnone
from .doctest import get_parent_func, relative_test_path, get_func_fq_name, DB_NAME

from nbconvert import HTMLExporter
from IPython.core import page
from IPython.core.display import display, Markdown, HTML

__all__ = ['show_test', 'doctest', 'find_dir_tests', 'lookup_db', 'find_test_matches', 'find_test_files', 'direct_test_match', 'fuzzy_test_match', 'get_pytest_html']

TestFunctionMatch = namedtuple('TestFunctionMatch', ['line_number', 'line'])

def show_test(elt)->str:
    "Show associated tests for a fastai function/class"
    md = ''.join(build_tests_markdown(elt))
    display(Markdown(md))

def doctest(elt):
    "Inline notebook popup for `show_test`"
    md = ''.join(build_tests_markdown(elt))
    output = HTMLExporter().markdown2html(md)
    try:    page.page({'text/html': output})
    except: display(Markdown(md))

def build_tests_markdown(elt):
    db_matches = [get_links(t) for t in lookup_db(elt)]
    try:
        direct, related = find_dir_tests(elt)
        direct = [get_links(t) for t in direct]
        related = [get_links(t) for t in related]
        direct = list(set(direct) - set(db_matches))
        related = list(set(related) - set(db_matches) - set(direct))
    except OSError as e:
        print('Could not find fastai/tests folder. If you installed from conda, please install developer build instead.')
        direct, related = [], []

    md = ''.join([
        tests2md(db_matches, 'This tests'),
        tests2md(direct, 'Direct tests'),
        tests2md(related, 'Related tests')
    ])
    fn_name = nbdoc.fn_name(elt)
    if len(md)==0: return f'No tests found for `{fn_name}`', md
    else: return f'Tests found for `{fn_name}`:', md

def tests2md(tests, type_label):
    if not tests: return ''
    md = [f'* `{cmd}` {link}' for link,cmd in tests]
    md = [f'\n\n{type_label}:'] + md
    return '\n'.join(md)

def get_pytest_html(elt, anchor_id:str, inline:bool=True)->Tuple[str,str]:
    title,body = build_tests_markdown(elt)
    htmlb = HTMLExporter().markdown2html(body).replace('\n','') # nbconverter fails to parse markdown if it has both html and '\n'
    htmlt = HTMLExporter().markdown2html(title).replace('\n','')
    anchor_id = anchor_id.replace('.', '-') + '-pytest'
    toggle_type = 'collapse' if inline else 'modal'
    link = f'<a class="source_link" data-toggle="{toggle_type}" data-target="#{anchor_id}" style="float:right; padding-right:10px">[test]</a>'
    body = get_pytest_card(htmlt, htmlb, anchor_id) if inline else get_pytest_modal(htmlt, htmlb, anchor_id)
    return link, body
    
def get_pytest_modal(title, body, anchor_id):
    "creates a modal html popup for `show_test`"
    return (f'<div class="modal" id="{anchor_id}" tabindex="-1" role="dialog"><div class="modal-dialog"><div class="modal-content">'
                f'<div class="modal-header">{title}<button type="button" class="close" data-dismiss="modal"></button></div>'
                f'<div class="modal-body">{body}</div>'
            '</div></div></div>')

def get_pytest_card(title, body, anchor_id):
    "creates a collapsible bootstrap card for `show_test`"
    return (f'<div class="collapse" id="{anchor_id}"><div class="card card-body"><div class="pytest_card">'
                f'{title+body}'
            '</div></div></div>'
            '<div style="height:1px"></div>') # hack to fix jumping bootstrap header

def lookup_db(elt)->List[Dict]:
    "Finds `this_test` entries from test_api_db.json"
    db_file = Path(abspath(join(dirname( __file__ ), '..')))/DB_NAME
    if not db_file.exists():
        print(f'Could not find {db_file}. Please make sure it exists at this location or run `make test`')
        return []
    with open(db_file, 'r') as f:
        db = json.load(f)
    key = get_func_fq_name(elt)
    return db.get(key, [])

def find_dir_tests(elt)->Tuple[List[Dict],List[Dict]]:
    "Searches `fastai/tests` folder for any test functions related to `elt`"
    test_dir = get_tests_dir(elt)
    test_files = find_test_files(elt)
    all_direct_matches = []
    all_fuzzy_matches = []
    for test_file in test_files:
        direct_matches, fuzzy_matches = find_test_matches(elt, test_file)
        all_direct_matches.extend(direct_matches)
        all_fuzzy_matches.extend(fuzzy_matches)
    return all_direct_matches, all_fuzzy_matches

def get_tests_dir(elt)->Path:
    "Absolute path of `fastai/tests` directory"
    fp = inspect.getfile(elt)
    fp.index('fastai/fastai')
    test_dir = Path(re.sub(r"fastai/fastai/.*", "fastai/tests", fp))
    if not test_dir.exists(): raise OSError('Could not find test directory at this location:', test_dir)
    return test_dir

def find_test_files(elt, exact_match:bool=False)->List[Path]:
    "Searches in `fastai/tests` directory for module tests"
    test_dir = get_tests_dir(elt)
    matches = [test_dir/o.name for o in os.scandir(test_dir) if _is_file_match(elt, o.name)]
    if len(matches) != 1:
        print('Could not find exact file match:', matches)
    return matches

def _is_file_match(elt, file_name:str, exact_match:bool=False):
    fp = inspect.getfile(elt)
    subdir = ifnone(_submodule_name(elt), '')
    exact_re = '' if exact_match else '\w*'
    return re.match(f'test_{subdir}\w*{Path(fp).stem}{exact_re}\.py', file_name)

def _submodule_name(elt)->str:
    "Returns submodule - utils, text, vision, imports, etc."
    if inspect.ismodule(elt): return None
    modules = elt.__module__.split('.')
    if len(modules) > 2:
        return modules[1]
    return None

def find_test_matches(elt, test_file:Path)->Tuple[List[Dict],List[Dict]]:
    "Find all functions in `test_file` related to `elt`"
    lines = get_lines(test_file)
    rel_path = relative_test_path(test_file)
    fn_name = get_qualname(elt) if not inspect.ismodule(elt) else ''

    direct_matches = direct_test_match(fn_name, lines, rel_path)
    fuzzy_matches = fuzzy_test_match(fn_name, lines, rel_path)
    return direct_matches, fuzzy_matches

def direct_test_match(fn_name:str, lines:List[Dict], rel_path:str)->List[TestFunctionMatch]:
    "Any `def test_function_name():` where test name contains function/class name"
    result = []
    fn_class,fn_name = separate_comp(fn_name)
    fn_class = '_'.join(fn_class)
    for idx,line in enumerate(lines):
        if re.match(f'\s*def test_\w*({fn_class}_)?{fn_name}\w*\(.*', line):
            result.append((idx,line))
    return [map_test(rel_path, lno, l) for lno,l in result]

def get_qualname(elt):
    return elt.__qualname__ if hasattr(elt, '__qualname__') else fn_name(elt)

def separate_comp(qualname:str):
    if not isinstance(qualname, str): qualname = get_qualname(qualname)
    parts = qualname.split('.')
    parts[-1] = remove_underscore(parts[-1])
    if len(parts) == 1: return [], parts[0]
    return parts[:-1], parts[-1]

def remove_underscore(fn_name):
    if fn_name and fn_name[0] == '_': return fn_name[1:] # remove private method underscore prefix
    return fn_name

def fuzzy_test_match(fn_name:str, lines:List[Dict], rel_path:str)->List[TestFunctionMatch]:
    "Find any lines where `fn_name` is invoked and return the parent test function"
    fuzzy_line_matches = _fuzzy_line_match(fn_name, lines)
    fuzzy_matches = [get_parent_func(lno, lines, ignore_missing=True) for lno,_ in fuzzy_line_matches]
    fuzzy_matches = list(filter(None.__ne__, fuzzy_matches))
    return [map_test(rel_path, lno, l) for lno,l in fuzzy_matches]

def _fuzzy_line_match(fn_name:str, lines)->List[TestFunctionMatch]:
    "Find any lines where `fn_name` is called"
    result = []
    _,fn_name = separate_comp(fn_name)
    for idx,line in enumerate(lines):
        if re.match(f'.*[\s\.\(]{fn_name}[\.\(]', line):
            result.append((idx,line))
    return result

def get_lines(file:Path)->List[str]:
    with open(file, 'r') as f: return f.readlines()

def map_test(test_file, line, line_text):
    "Creates dictionary test format to match doctest api"
    test_name = re.match(f'\s*def (test_\w*)', line_text).groups(0)[0]
    return { 'file': test_file, 'line': line, 'test': test_name }

def get_links(metadata)->Tuple[str,str]:
    "Returns source code link and pytest command"
    return nbdoc.get_source_link(**metadata), pytest_command(**metadata)

def pytest_command(file:str, test:str, **kwargs)->str:
    "Returns CLI command to run specific test function"
    return f'pytest -sv {file}::{test}'