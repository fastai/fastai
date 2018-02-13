
# coding: utf-8

"""
Setup script for installing fastai
"""

##########################################################################
## Imports
##########################################################################

#from distutils.core import setup
from setuptools import setup


##########################################################################
## Setup
##########################################################################

setup(
    name = "fastai",
    packages = ['fastai', 'fastai/models', 'fastai/models/cifar10'],
    version = 0.6 ,
    description = "The fast.ai deep learning and machine learning library. Git pull fastai, for all fast.ai sessions and tutorials also",
    author = "Jeremy Howard and contributors",
    author_email = "info@fast.ai",
    license = "Apache License 2.0",
    url = "https://github.com/fastai/fastai",
    download_url =  'https://github.com/fastai/fastai/archive/0.6.tar.gz',
    install_requires =
     ['awscli', 'bcolz', 'bleach', 'certifi', 'cycler', 'decorator', 'entrypoints', 'feather-format', 'graphviz', 'html5lib',
      'ipykernel', 'ipython', 'ipython-genutils', 'ipywidgets', 'isoweek', 'jedi', 'Jinja2', 'jsonschema', 'jupyter',
      'jupyter-client', 'jupyter-console', 'jupyter_contrib_nbextensions', 'jupyter-core', 'kaggle-cli', 'MarkupSafe',
      'matplotlib', 'mistune', 'nbconvert', 'nbformat', 'notebook', 'numpy', 'olefile', 'opencv-python', 'pandas',
      'pandas_summary', 'pandocfilters', 'pexpect', 'pickleshare', 'Pillow', 'plotnine', 'prompt-toolkit',
      'ptyprocess', 'Pygments', 'pyparsing', 'python-dateutil', 'pytz', 'PyYAML', 'pyzmq', 'qtconsole', 'scipy',
      'seaborn', 'simplegeneric', 'six', 'sklearn_pandas', 'terminado', 'testpath', 'torchtext', 'tornado', 'tqdm',
      'traitlets', 'wcwidth', 'webencodings', 'widgetsnbextension'],
    keywords = ['deeplearning', 'pytorch', 'machinelearning'],
    classifiers = ['Development Status :: 3 - Alpha', 'Programming Language :: Python', 'Programming Language :: Python :: 3.6']
)

