
# coding: utf-8

""" Setup script for installing fastai """

#from distutils.core import setup
from setuptools import setup

setup(
    name = "fastai",
    packages = ['fastai', 'fastai/models', 'fastai/models/cifar10'],
    version = '0.7.0',
    description = "The fastai deep learning and machine learning library.",
    author = "Jeremy Howard and contributors",
    author_email = "info@fast.ai",
    license = "Apache License 2.0",
    url = "https://github.com/fastai/fastai",
    download_url =  'https://github.com/fastai/fastai/archive/0.7.0.tar.gz',
    install_requires =
     ['bcolz', 'bleach', 'certifi', 'cycler', 'decorator', 'entrypoints', 'feather-format', 'graphviz', 'html5lib',
      'ipykernel', 'ipython', 'ipython-genutils', 'ipywidgets', 'isoweek', 'jedi', 'Jinja2', 'jsonschema', 'jupyter',
      'MarkupSafe', 'matplotlib', 'numpy', 'opencv-python', 'pandas',
      'pandas_summary', 'pickleshare', 'Pillow', 'plotnine',
      'ptyprocess', 'Pygments', 'pyparsing', 'python-dateutil', 'pytz', 'PyYAML', 'pyzmq', 'scipy',
      'seaborn', 'simplegeneric', 'sklearn_pandas', 'spacy', 'testpath', 'torch<0.4', 'torchtext',
      'torchvision', 'tornado', 'tqdm', 'traitlets', 'wcwidth', 'webencodings', 'widgetsnbextension'],
    keywords = ['deeplearning', 'pytorch', 'machinelearning'],
    classifiers = ['Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
