# fast.ai [![Build Status](https://travis-ci.org/fastai/fastai.svg?branch=master)](https://travis-ci.org/fastai/fastai)
The fast.ai deep learning library, lessons, and tutorials.

Copyright 2017 onwards, Jeremy Howard. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.

## Current Status
This is an alpha version.

Most of the library is quite well tested since many students have used it to complete the [Practical Deep Learning for Coders](http://course.fast.ai) course. However it hasn't been widely used yet outside of the course, so you may find some missing features or rough edges.

If you're interested in using the library in your own projects, we're happy to help support any bug fixes or feature additions you need&mdash;please use [http://forums.fast.ai](http://forums.fast.ai) to discuss.

## To install

### Prerequisites
* [Anaconda](https://conda.io/docs/user-guide/install/index.html#), manages Python environment and dependencies

### Normal installation
1. Download project: `git clone https://github.com/fastai/fastai.git`
1. Move into root folder: `cd fastai`
1. Set up Python environment: `conda env update`
1. Activate Python environment: `conda activate fastai`
    - If this fails, use instead: `source activate fastai`

### Install as pip package

Do not use this method anymore, as it'll install `fastai-1.x` instead. (
<s>`pip install git+https://github.com/fastai/fastai.git`</s>)

If you don't want to use `conda`, but to install via `pip`, either do:
1. Download project: `git clone https://github.com/fastai/fastai.git`
1. Move into root folder: `cd fastai`
1. Move into fastai-0.7 folder: `cd old`
1. Install fastai-0.7: `pip install .`

Or alternatively, this can be done directly from github: `pip install "git+https://github.com/fastai/fastai#egg=fastai&subdirectory=old"` (the `subdirectory=old` is what telling pip to use `setup.py` under `old`.)

### CPU only environment
Use this if you do not have an NVidia GPU. Note you are encouraged to use Paperspace to access a GPU in the cloud by following this [guide](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/paperspace.md).

`conda env update -f environment-cpu.yml`

Anytime the instructions say to activate the Python environment, run `conda activate fastai-cpu` or `source activate fastai-cpu`.

## To update
1. Update code: `git pull`
1. Update dependencies: `conda env update`

## To test
Before submitting a pull request, run the unit tests:

1. Activate Python environment: `conda activate fastai`
    - If this fails, use instead: `source activate fastai`
1. Run tests: `pytest tests`

### To run specific test file
1. Activate Python environment: `conda activate fastai`
    - If this fails, use instead: `source activate fastai`
1. `pytest tests/[file_name.py]`

### If tests fail
The `master` build should always be clean and pass. If `master` isn't passing, try the following:

* make sure the virtual environment is activated with `conda activate fastai` or `source activate fastai`
* update the project (see above section)
* consider using the cpu environment if testing on a computer without a GPU (see above section)

If the tests are still failing on `master`, please [file an issue on GitHub](https://github.com/fastai/fastai/issues) explaining the issue and steps to reproduce the problem.

If the tests are failing on your new branch, but they pass on `master`, this means your code changes broke one of the tests. Investigate what might be causing this and play around until you get the test passing. Feel free to ask for help!
