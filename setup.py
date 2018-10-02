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
version = '1.0.3'
create_version_file(version)

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

def to_list(buffer): return list(filter(None, buffer.splitlines()))

# XXX: require torch>=1.0.0 once it's released, for now get the user to install it explicitly
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
torchvision>=0.2.1
typing
""")

if sys.version_info < (3,7): requirements.append('dataclasses')

# optional requirements to skip for now:
# cupy - is only required for QRNNs - sgguger thinks later he will get rid of this dep.
# fire - will be eliminated shortly
# nbconvert
# nbformat
# traitlets
# jupyter_contrib_nbextensions

setup_requirements = to_list("""
pytest-runner
""")

test_requirements = to_list("""
pytest
torch>=0.4.9
torchvision>=0.2.1
numpy>=1.12
""")

# list of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
setup(
    author="Jeremy Howard",
    author_email='info@fast.ai',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="fastai makes deep learning with PyTorch faster, more accurate, and easier",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='fastai',
    name='fastai',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    python_requires='>=3.6',
    url='https://github.com/fastai/fastai',
    version=version,
    zip_safe=False,
)
