#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import re
from setuptools import setup, find_packages

# note: version is maintained inside fastai/version.py
exec(open('fastai/version.py').read())

with open('README.md') as readme_file: readme = readme_file.read()

# helper functions to make it easier to list dependencies not as a python list, but vertically w/ optional built-in comments to why a certain version of the dependency is listed
def cleanup(x):
    x = x.strip()               # whitespace
    x = re.sub(r' *#.*', '', x) # comments
    return x
def to_list(buffer): return list(filter(None, map(cleanup, buffer.splitlines())))

### normal dependencies ###
#
# IMPORTANT: when updating these, please make sure to sync conda/meta.yaml and docs/install.md (the "custom dependencies" section)
#
# these get resolved and installed via either of these two:
#
#   pip install fastai
#   pip install -e .
#
# dependencies to skip for now:
# - cupy - is only required for QRNNs - sgugger thinks later he will get rid of this dep.
#
# XXX: when spacy==2.0.18 is on anaconda channel, put it in place (it's already on pypi) and remove its deps: cymem, regex, thinc (and update meta.yaml with the same)
requirements = to_list("""
    bottleneck           # performance-improvement for numpy
    cymem==2.0.2         # remove once spacy==2.0.18 is on anaconda channel
    dataclasses ; python_version<'3.7'
    fastprogress>=0.1.18
    matplotlib
    numexpr              # performance-improvement for numpy
    numpy>=1.12
    nvidia-ml-py3
    pandas
    packaging
    Pillow
    pyyaml
    regex==2018.01.10    # remove once spacy==2.0.18 is on anaconda channel
    requests
    scipy
    spacy==2.0.16
    thinc==6.12.0        # remove once spacy==2.0.18 is on anaconda channel
    torch
    torchvision
    typing
""")

### developer dependencies ###
#
# anything else that's not required by a user to run the library, but
# either is an enhancement or a developer-build requirement goes here.
#
# the [dev] feature is documented here:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
#
# these, including the normal dependencies, get installed with:
#
#   pip install fastai[dev]
#
# or via an editable install:
#
#   pip install -e .[dev]
#
# some of the listed modules appear in test_requirements as well, as explained below.
#
dev_requirements = { 'dev' : to_list("""
    coverage
    distro
    ipython
    jupyter
    jupyter_contrib_nbextensions
    nbconvert>=5.4
    nbdime                       # help with nb diff/merge
    nbformat
    notebook>=5.7.0
    pip>=9.0.1
    pipreqs>=0.4.9
    pytest
    traitlets
    wheel>=0.30.0
""") }

### setup dependencies ###
# need at least setuptools>=36.2 to support syntax:
#   dataclasses ; python_version<'3.7'
setup_requirements = to_list("""
    pytest-runner
    setuptools>=36.2
""")

# notes:
#
# * these deps will be installed locally under .eggs/ and will not be
#   visible to pytest unless it's invoked via `python setup test`.
#   Therefore it's the best to install them explicitly with:
#   pip install -e .[dev]
#
### test dependencies ###
test_requirements = to_list("""
    pytest
""")

# list of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
setup(
    name = 'fastai',
    version = __version__,

    packages = find_packages(),
    include_package_data = True,

    install_requires = requirements,
    setup_requires   = setup_requirements,
    extras_require   = dev_requirements,
    tests_require    = test_requirements,
    python_requires  = '>=3.6',

    test_suite = 'tests',

    description = "fastai makes deep learning with PyTorch faster, more accurate, and easier",
    long_description = readme,
    long_description_content_type = 'text/markdown',
    keywords = 'fastai, deep learning, machine learning',

    license = "Apache Software License 2.0",

    url = 'https://github.com/fastai/fastai',

    author = "Jeremy Howard",
    author_email = 'info@fast.ai',

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    zip_safe = False,
)
