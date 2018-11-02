#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from pathlib import Path
from setuptools import setup, find_packages

# note: version is maintained inside fastai/version.py
exec(open('fastai/version.py').read())

with open('README.md') as readme_file:   readme = readme_file.read()

def to_list(buffer): return list(filter(None, map(str.strip, buffer.splitlines())))

### normal dependencies ###
#
# these get resolved and installed via either of these two:
#
#   pip install fastai
#   pip install -e .
#
# XXX: require torch>=1.0.0 once it's released, for now get the user to install it explicitly
# XXX: using a workaround for torchvision, once torch-1.0.0 is out and a new torchvision depending on it is released switch to torchvision>=0.2.2
# XXX: temporarily pinning spacy and its dependencies (regex, thinc, and cymem) to have a stable environment during the course duration.
requirements = to_list("""
    fastprogress>=0.1.15
    ipython
    jupyter
    matplotlib
    notebook>=5.7.0
    nbconvert
    nbformat
    numpy>=1.12
    pandas
    Pillow
    requests
    scipy
    spacy==2.0.16
    regex
    thinc==6.12.0
    cymem==2.0.2
    torchvision-nightly
    traitlets
    typing
    pyyaml
""")

# dependencies to skip for now:
#
# cupy - is only required for QRNNs - sgguger thinks later he will get rid of this dep.

if sys.version_info < (3,7): requirements.append('dataclasses')

### developer dependencies ###
#
# anything else that's not required by a user to run the library, but
# either an enhancement or developer-build requirement goes here.
#
# the [dev] feature is documented here:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
#
# these get installed with:
#
#   pip install -e .[dev]
#
# some of the listed modules appear in test_requirements as well, explained below.
#
dev_requirements = { 'dev' : to_list("""
    coverage
    distro
    jupyter_contrib_nbextensions
    pip>=9.0.1
    pipreqs>=0.4.9
    pytest
    wheel>=0.30.0
""") }

### setup dependencies ###
setup_requirements = to_list("""
    pytest-runner
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
