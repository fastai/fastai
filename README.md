# fast.ai
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
You can also install this library in the local environment using `pip`

`pip install fastai`

However this is not currently the recommended approach, since the library is being updated much more frequently than the pip release, fewer people are using and testing the pip version, and pip needs to compile many libraries from scratch (which can be slow). 

### CPU only environment
Use this if you do not have an NVidia GPU. Note you are encouraged to use Paperspace to access a GPU in the cloud by following this [guide](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/paperspace.md).

`conda env update -f environment-cpu.yml`

## To update
To update everything at any time:

1. Update code: `git pull`
1. Update dependencies: `conda env update`

## To test
Before submitting a pull request, run the unit tests:

1. Activate Python environment: `conda activate fastai`
    - If this fails, use instead: `source activate fastai`
1. Run tests: `python -m pytest tests`

### To run specific test file
1. Activate Python environment: `conda activate fastai`
    - If this fails, use instead: `source activate fastai`
1. `python -m pytest tests/[file_name.py]`
