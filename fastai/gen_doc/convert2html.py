import os.path, re, nbformat, jupyter_contrib_nbextensions
from nbconvert.preprocessors import Preprocessor
from nbconvert import HTMLExporter
from traitlets.config import Config
from pathlib import Path

__all__ = ['read_nb', 'convert_nb', 'convert_all']

class HandleLinksPreprocessor(Preprocessor):
    "A preprocesser that replaces all the .ipynb by .html in links. "
    def preprocess_cell(self, cell, resources, index):
        if 'source' in cell and cell.cell_type == "markdown":
            cell.source = re.sub(r"\((.*)\.ipynb(.*)\)",r"(\1.html\2)",cell.source).replace('¶','')

        return cell, resources

exporter = HTMLExporter(Config())
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
#Loads the template to deal with hidden cells.
exporter.template_file = 'jekyll.tpl'
path = Path(__file__).parent
exporter.template_path.append(str(path))
#Preprocesser that converts the .ipynb links in .html
#exporter.register_preprocessor(HandleLinksPreprocessor, enabled=True)

def read_nb(fname):
    "Read the notebook in `fname`."
    with open(fname,'r') as f: return nbformat.reads(f.read(), as_version=4)

def convert_nb(fname, dest_path='.'):
    "Convert a notebook `fname` to html file in `dest_path`."
    from .gen_notebooks import remove_undoc_cells
    nb = read_nb(fname)
    nb['cells'] = remove_undoc_cells(nb['cells'])
    fname = Path(fname)
    dest_name = fname.with_suffix('.html').name
    meta = nb['metadata']
    meta_jekyll = meta['jekyll'] if 'jekyll' in meta else {'title': fname.with_suffix('').name}
    with open(f'{dest_path}/{dest_name}','w') as f:
        f.write(exporter.from_notebook_node(nb, resources=meta_jekyll)[0])

def convert_all(folder, dest_path='.'):
    "Convert all notebooks in `folder` to html pages in `dest_path`."
    path = Path(folder)
    nb_files = path.glob('*.ipynb')
    for file in nb_files: convert_nb(file, dest_path=dest_path)
