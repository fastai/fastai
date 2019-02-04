#!/usr/bin/env python3

import io, sys, argparse, json

if sys.stdin: input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--textconv', action="store_true", help="Print results to output")
parser.add_argument('-d', '--doc-mode', action="store_true", help="fastai docs nb-specific strip out")
parser.add_argument('files', nargs='*', help='Files to strip output from')
args = parser.parse_args()

# define which fields need to be kept:
cell_metadata_keep_code = []
cell_metadata_keep_docs = ['hide_input']
nb_metadata_keep        = ['kernelspec', 'jekyll']

def clean_cell_outputs(o):
    if 'execution_count' in o: o['execution_count'] = None

### filter for doc nb cells ###
# 1. reset execution_count (in cell and cell's outputs field)
# 2. keep only cell_metadata_keep_doc fields

def clean_cell_docs(o):
    if 'execution_count' in o: o['execution_count'] = None
    if 'outputs' in o:
        for l in o['outputs']: clean_cell_outputs(l)

    o['metadata'] = { k:o['metadata'][k] for k in o['metadata'].keys() if k in cell_metadata_keep_docs }
    return o

### filter for code nb cells ###
# 1. reset execution_count
# 2. delete cell's metadata
# 3. delete cell's outputs

def clean_cell_code(o):
    if 'execution_count' in o: o['execution_count'] = None
    if 'outputs'         in o: o['outputs']         = []
    o['metadata'] = {}
    return o

# optimize runtime
clean_cell = clean_cell_code if not args.doc_mode else clean_cell_docs

### filter for nb top level entries ###
# 1. keep only nb_metadata_keep fields
# 2. the other rules apply based on clean_cell alias

def clean_nb(s):
    s['cells']    = [ clean_cell(o) for o in s['cells'] ]
    s['metadata'] = { k:s['metadata'][k] for k in s['metadata'].keys() if k in nb_metadata_keep }

for filename in args.files:
    if not filename.endswith('.ipynb'): continue
    with io.open(filename, 'r', encoding='utf-8') as f: s = json.load(f)
    clean_nb(s)
    x = json.dumps(s, sort_keys=True, indent=1, ensure_ascii=False)

    if args.textconv:
        # XXX: if there is more than one file, this is probably wrong
        output_stream.write(x)
        output_stream.write("\n")
        output_stream.flush()
    else:
        with io.open(filename, 'w', encoding='utf-8') as f:
            f.write(x)
            f.write("\n")

# implied textconv mode
if not args.files and input_stream:
    s = json.load(input_stream)
    clean_nb(s)
    x = json.dumps(s, sort_keys=True, indent=1, ensure_ascii=False)
    output_stream.write(x)
    output_stream.write("\n")
    output_stream.flush()

