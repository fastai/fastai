#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from pathlib import Path
from setuptools import setup, find_packages

def create_version_file(version):
    print('-- Building version ' + version)
    version_path = Path.cwd() / 'fastai' / 'version.py'
    with open(version_path, 'w') as f:
        f.write("__all__ = ['__version__']\n")
        f.write("__version__ = '{}'\n".format(version))

# version
version = '1.0.5.dev0'
create_version_file(version)

with open('README.md') as readme_file:   readme = readme_file.read()
with open('CHANGES.md') as history_file: history = history_file.read()

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
requirements = to_list("""
    fastprogress>=0.1.9
    ipython
    jupyter
    matplotlib
    numpy>=1.12
    pandas
    Pillow
    requests
    scipy
    spacy
    torchvision-nightly
    typing
""")

# dependencies to skip for now:
#
# cupy - is only required for QRNNs - sgguger thinks later he will get rid of this dep.
# fire - will be eliminated shortly

if sys.version_info < (3,7): requirements.append('dataclasses')

### developer dependencies ###
#
# anything else that's not required by a user to run the library, but
# either an enhancement or developer-build requirement goes here.
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
#
# these get installed with:
#
#   pip install -e .[dev]
#
dev_requirements = { 'dev' : to_list("""
    bumpversion==0.5.3
    distro
    gputil
    jupyter_contrib_nbextensions
    nbconvert
    nbformat
    pip>=18.1
    pipreqs>=0.4.9
    traitlets
    wheel>=0.30.0
""") }

### setup dependencies ###
setup_requirements = to_list("""
    pytest-runner
""")

### test dependencies ###
test_requirements = to_list("""
    pytest
    pytest-pspec
""")

# list of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
setup(
    name = 'fastai',
    version = version,

    packages = find_packages(),
    include_package_data = True,

    install_requires = requirements,
    setup_requires   = setup_requirements,
    extras_require   = dev_requirements,
    tests_require    = test_requirements,
    python_requires  = '>=3.6',

    test_suite = 'tests',

    description = "fastai makes deep learning with PyTorch faster, more accurate, and easier",
    long_description = readme + '\n\n' + history,
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
