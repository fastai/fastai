#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import re
from setuptools import setup, find_packages

from distutils.core import Command

class DepsCommand(Command):
    """A custom distutils command to print selective dependency groups.

    # show available dependency groups:
    python setup.py -q deps

    # print dependency list for specified groups
    python setup.py -q deps --dep-groups=core,vision

    # see all options:
    python setup.py -q deps --help
    """

    description = 'show dependency groups and their packages'
    user_options = [
        # format: (long option, short option, description).
        ('dep-groups=', None, 'comma separated dependency groups'),
        ('dep-quote',   None, 'quote each dependency'),
        ('dep-conda',   None, 'adjust output for conda'),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.dep_groups = ''
        self.dep_quote = False
        self.dep_conda = False

    def finalize_options(self):
        """Post-process options."""
        pass

    def parse(self):
        arg = self.dep_groups.strip()
        return re.split(r' *, *', arg) if len(arg) else []

    def run(self):
        """Run command."""
        wanted_groups = self.parse()

        deps = []
        invalid_groups = []
        for grp in wanted_groups:
            if grp in dep_groups: deps.extend(dep_groups[grp])
            else:                 invalid_groups.append(grp)

        if invalid_groups or not wanted_groups:
            print("Available dependency groups:", ", ".join(sorted(dep_groups.keys())))
            if invalid_groups:
                print(f"Error: Invalid group name(s): {', '.join(invalid_groups)}")
                exit(1)
        else:
            # prepare for shell word splitting (no whitespace in items)
            deps = [re.sub(" ", "", x, 0) for x in sorted(set(deps))]
            if self.dep_conda:
                for i in range(len(deps)):
                    # strip pip-specific syntax
                    deps[i] = re.sub(r';.*',     '',         deps[i])
                    # rename mismatching package names
                    deps[i] = re.sub(r'^torch>', 'pytorch>', deps[i])
            if self.dep_quote:
                # for manual copy-n-paste (assuming no " in vars)
                print(" ".join(map(lambda x: f'"{x}"', deps)))
            else:
                # if fed directly to `pip install` via backticks/$() don't quote
                print(" ".join(deps))

# note: version is maintained inside fastai/version.py
exec(open('fastai/version.py').read())

with open('README.md') as readme_file: readme = readme_file.read()

# helper functions to make it easier to list dependencies not as a python list, but vertically w/ optional built-in comments to why a certain version of the dependency is listed
def cleanup(x): return re.sub(r' *#.*', '', x.strip()) # comments
def to_list(buffer): return list(filter(None, map(cleanup, buffer.splitlines())))

### normal dependencies ###
#
# these get resolved and installed via either of these two:
#
#   pip install fastai
#   pip install -e .
#
# IMPORTANT: when updating these, please make sure to sync conda/meta.yaml
dep_groups = {
    'core':   to_list("""
        bottleneck           # performance-improvement for numpy
        dataclasses ; python_version<'3.7'
        fastprogress>=0.2.1
        beautifulsoup4
        matplotlib
        numexpr              # performance-improvement for numpy
        numpy>=1.15
        nvidia-ml-py3
        pandas
        packaging
        Pillow
        pyyaml
        pynvx>=1.0.0 ; platform_system=="Darwin"  # only pypi at the moment
        requests
        scipy
        torch>=1.0.0
"""),
    'text':   to_list("""
        spacy>=2.0.18; python_version<'3.8'
"""),
    'vision': to_list("""
        torchvision
"""),
}

requirements = [y for x in dep_groups.values() for y in x]

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
#   pip install "fastai[dev]"
#
# or via an editable install:
#
#   pip install -e ".[dev]"
#
# some of the listed modules appear in test_requirements as well, as explained below.
#
dev_requirements = { 'dev' : to_list("""
    coverage                     # make coverage
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
    pytest>=4.4.0
    pytest-xdist                 # make test-fast (faster parallel testing)
    responses                    # for requests testing
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
    cmdclass = { 'deps': DepsCommand },

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
