from fastai.gen_doc import nbdoc
from pathlib import Path
import inspect
import re
import os
from nbconvert import HTMLExporter
from IPython.core import page
from IPython.core.display import display, Markdown, HTML

def show_doctest(elt, markdown=True):
    fn_name = nbdoc.fn_name(elt)
    md = f'Tests found for `{fn_name}`:'
    duplicates = set()
    dms, fms = get_tests(elt)
    dm_str = ''
    for link,cmd in dms:
        if link in duplicates: continue
        dm_str += f'\n * `{cmd}` [\[source\]]({link})'
        duplicates.add(link)
    if dm_str: md += '\n\nDirect Tests:' + dm_str
        
    
    fm_str = ''
    for link,cmd in fms:
        if link in duplicates: continue
        fm_str += f'\n * `{cmd}` [\[source\]]({link})'
        duplicates.add(link)
    if fm_str: md += '\n\nAncillary Tests:' + fm_str
    md += f'\n\nSkeleton test:\n```\n{create_skeleton_test(elt)}\n```'
    if markdown: display(Markdown(md))
    else: return md

def doctest(elt):
    md = show_doctest(elt, markdown=False)
    output = HTMLExporter().markdown2html(md)
    try:    page.page({'text/html': output})
    except: display(Markdown(md))

def get_tests(elt):
    test_dir = get_test_dir(elt)
    test_files = get_test_files(elt)
    all_direct_matches = []
    all_fuzzy_matches = []
    for test_file in test_files:
        direct_matches, fuzzy_matches = find_test_lines(elt, test_file)
        for lno,l in direct_matches: all_direct_matches.append(get_links(lno,l,test_file))
        for lno,l in fuzzy_matches: all_fuzzy_matches.append(get_links(lno,l,test_file))
    return all_direct_matches, all_fuzzy_matches

def get_test_dir(elt):
    fp = inspect.getfile(elt)
    fp.index('fastai/fastai')
    test_dir = Path(re.sub(r"fastai/fastai/.*", "fastai/tests", fp))
    if not test_dir.exists(): raise Exception('Could not find test path at this location:', test_dir)
    return test_dir

def get_test_files(elt):
    test_dir = get_test_dir(elt)
    fp = inspect.getfile(elt)
    def is_match(file_name):
        sub_dir = submodule(elt)
        if sub_dir is not None:
            return re.match(f'test_{sub_dir}\w*{Path(fp).stem}\w*\.py', file_name)
        return re.match(f'test\w*{Path(fp).stem}\w*\.py', file_name)
    matches = [test_dir/o.name for o in os.scandir(test_dir) if is_match(o.name)]
    if len(matches) != 1: 
        print('Could not find exact file match:', matches)
    return matches

def submodule(elt):
    modules = elt.__module__.split('.')
    if len(modules) > 2:
        return modules[1]
    return None

def find_direct_test_function_match(elt, lines):
    fn_name = nbdoc.fn_name(elt)
    result = []
    for idx,line in enumerate(lines):
        if re.match(f'^def test_\w*{fn_name}\w*\(.*', line):
            result.append((idx,line))
    return result

def _fuzzy_line_match(elt, lines):
    fn_name = nbdoc.fn_name(elt)
    result = []
    for idx,line in enumerate(lines):
        if re.match(f'.*[\s\.\(]{fn_name}[\.\(]', line):
            result.append((idx,line))
    return result

def _get_parent_func(lineno, lines):
    for idx,l in enumerate(reversed(lines[:lineno])):
        if re.match(f'^def test', l):
            return (lineno - (idx+1)), l
    return None

def fuzzy_match(elt, lines):
    fuzzy_line_matches = _fuzzy_line_match(elt, lines)
    fuzzy_matches = [_get_parent_func(lineno, lines) for lineno,line in fuzzy_line_matches]
    fuzzy_matches = list(filter(None.__ne__, fuzzy_matches))
    return fuzzy_matches

def find_test_lines(elt, test_file):
    lines = get_lines(test_file)
    direct_matches = find_direct_test_function_match(elt, lines)
    fuzzy_matches = fuzzy_match(elt, lines)
    return direct_matches, fuzzy_matches

def get_lines(file):
    with open(file, 'r') as f:
        return f.readlines()

def relative_test_path(test_file):
    return '/'.join(test_file.parts[-2:])

def get_links(lno, line, test_file):
    return _source_link(lno, test_file), _pytest_command(line, test_file)

def _pytest_command(line, test_file):
    test_name = re.match(f'^def (test\w*)', line).groups(0)[0]
    test_path = relative_test_path(test_file)
    return f'pytest -sv -k {test_name} {test_path}'

def _source_link(lno, test_file):
    test_path = relative_test_path(test_file)
    return f'{nbdoc.SOURCE_URL}{test_path}#L{lno}'

def create_skeleton_test(elt):
    fn_name = nbdoc.fn_name(elt)
    return f'def test_{fn_name}():\n\tpass'