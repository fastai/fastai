#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import ast
import re
import contextlib
from pathlib import Path
import subprocess
from .templates import *
import fire


def get_cls_str(ps):
    cls_name = ps[0]
    return f"Class {cls_name}"

def get_sub_arg(ps):
    arg_name, arg_type, arg_default = ''.join(ps).split(',', 2)
    if arg_type and arg_default:
        return f'*{arg_name}* (type {arg_type}, default {arg_default})'
    elif arg_type:
        return f'*{arg_name}* (type {arg_type})'
    elif arg_default:
        return f'*{arg_name}* (default {arg_default})'
    else:
        return f'*{arg_name}*'

def get_xref_str(ps):
    xref_id, xref_cap = ps if len(ps) == 2 else ps*2
    return f"xref:{xref_id}[{xref_cap}]"

def get_method_str(ps):
    method_name, doc_string = ''.join(ps).split(',', 1)
    result = f'*{method_name}*'

    if doc_string:
        result += f':: {doc_string}' if doc_string else ''
    return result

def parse_tmpl(s):
    inner = s.group(1)
    fn_name,*params = inner.split(' ', 1)
    fn = _fn_lu[fn_name]
    return fn(params)

def parse_module(file_path):
    module = ast.parse(file_path.read_text())
    tmpl_str = HEADER.format(file_path.name.rsplit('.',1)[0])
    cls_defs = [node for node in module.body if isinstance(node, ast.ClassDef)]
    mod_func_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    for cls_def in cls_defs:
        cls_name = cls_def.name
        cls_bases = ','.join([parse(each) for each in cls_def.bases])
        tmpl_str += f'== {{{{class {cls_name}{":" + cls_bases if cls_bases else ""}}}}}\n\n'
        method_str = None
        for fn_def in (fn_def for fn_def in cls_def.body if isinstance(fn_def, ast.FunctionDef)):
            if fn_def.name == '__init__':
                tmpl_str += "=== Arguments\n" + parse_args(fn_def.args) + "\n\n"
            else:
                if not method_str:
                    method_str = '=== Methods\n\n'
                doc_str = ast.get_docstring(fn_def)
                method_str += f'{{{{method {fn_def.name},{doc_str if doc_str else ""}}}}}\n\n'
        tmpl_str += method_str if method_str else ''
    method_str = None
    for fn_def in mod_func_defs:
        if not method_str:
            method_str = '== Module Functions\n\n'
        doc_str = ast.get_docstring(fn_def)
        method_str += f'{{{{method {fn_def.name},{doc_str if doc_str else ""}}}}}\n\n'
    tmpl_str += method_str if method_str else ''
    return tmpl_str

def parse_args(args):
    arg_strs = [f'{arg.arg},{arg.annotation.id if arg.annotation else ""}' for arg in args.args if arg.arg != 'self']
    defaults = parse_defaults(args.defaults)
    defaults = [None]*(len(arg_strs)-len(defaults)) + defaults
    return '\n\n'.join(['{{' + f'arg {arg},{default if default else ""}' + '}}' for arg, default in zip(arg_strs, defaults)])

def parse_defaults(defs):
    return [parse(each) for each in defs]

def parse_num(o):
    return str(o.n)

def parse_str(o):
    return o.s

def parse_call(o):
    return o.func.id + '()'

def parse(o):
    return _parser_dict.get(type(o), lambda x: str(x))(o)

@contextlib.contextmanager
def working_directory(path):
    prev_cwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev_cwd))

def gen_ascii_docs(src='fastai'):
    """Generate documentation for fastai library in HTML (asciidoctor required)
    :param str src: The absolute/relative path of source file/dir
    """
    os.chdir(Path(__file__).absolute().parent)
    with working_directory('..'):
        path = Path(src)
        if path.is_dir():
            file_paths = list(path.glob('**/*.py'))
        else:
            file_paths = [path]

    pat = re.compile('^(?!__init__).*.py\Z')
    for file_path in file_paths:
        if pat.match(file_path.name):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with working_directory('..'):
                tmpl_str = parse_module(file_path)

            (file_path.parent/(file_path.name.rsplit('.',1)[0] + '.adoc.tmpl')).write_text(tmpl_str)
            (file_path.parent/(file_path.name.rsplit('.',1)[0] + '.adoc')).write_text(re.sub(r"{{(.*?)}}", parse_tmpl, tmpl_str, flags=re.DOTALL))
    if path.is_dir():
        subprocess.call(['asciidoctor', str(path) + '/**/*.adoc'])
    else:
        subprocess.call(['asciidoctor', str(path).rsplit('.',1)[0] + '.adoc'])


_fn_lu = {
    'class': get_cls_str,
     'arg': get_sub_arg,
     'xref': get_xref_str,
     'method': get_method_str
}

_parser_dict = {
    ast.Dict:lambda x: '{}',
    ast.arguments: parse_args,
    ast.Call: parse_call,
    ast.Num: parse_num,
    ast.Str: parse_str,
    ast.Name: lambda x: x.id,
    ast.NameConstant: lambda x: str(x.value),
    ast.Attribute: lambda x: x.attr,
    ast.List: lambda x: '[]',
    list: lambda x: list(map(parse, x))
}

if __name__ == '__main__':
    fire.Fire(gen_ascii_docs)