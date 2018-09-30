# fastai

The fastai deep learning library.

Copyright 2017 onwards, fast.ai, Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.

## Current Status

This is a ground-up rewrite of fastai. Everything should work, although docs are still in progress. If you're interested in contributing, join the discussion at: http://forums.fast.ai/c/fastai-dev.

## Install

To use the notebooks or the beta version of the fastai modules you will need:
- to use python 3.7 or python 3.6 with [dataclasses](https://github.com/ericvsmith/dataclasses): `pip install dataclasses`
- to use the `pytorch-nightly` conda package, or the master branch of [pytorch master](https://github.com/pytorch/pytorch#from-source)
- to install [fastprogress](https://github.com/fastai/fastprogress): `pip install fastprogress`

### PyPI Install

First install the nightly `pytorch` build, e.g. for CUDA 9.2:

    pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html

If you have a different CUDA version find the right build [here](https://pytorch.org/get-started/locally/). Choose Preview/Linux/Pip/python3.6|python3.7 and Your CUDA version and it will give you the correct install instruction.

Now you can install `fastai`. Note, that this is a beta test version at the moment, please [report any issues](https://github.com/fastai/fastai_pytorch/issues/):

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url  https://pypi.org/simple/ fastai==1.0.0b6

 Sometimes, the last `pip` command still tries to get `torch-0.4.1`. If that happens to you, do:

    pip uninstall torchvision fastai
    pip install --no-deps torchvision
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url  https://pypi.org/simple/ fastai==1.0.0b6



### Conda Install

To install fastai with CUDA 9.2 simply run (read the paragraph after this for other GPU and CPU options):

    conda install -c pytorch -c fastai/label/test fastai cuda92

If your setup doesn't have CUDA support remove the `cuda92` above.

For different versions of the CUDA toolkit, you'll need to install the appropriate CUDA conda package based on what you've got installed on your system (i.e. instead of `cuda92` in the above, pick the appropriate option for whichever toolkit version you have installed; to see a list of options type: `conda search "cuda*" -c pytorch`).

Note, that this is a beta test version at the moment, please [report any issues](https://github.com/fastai/fastai_pytorch/issues/). We are currently using a re-packaged torchvision in order to support pytorch-nightly, which is required for using fastai.

### Developer Install

First, follow the instructions above for either `PyPi` or `Conda`. Then remove the fastai package (`pip uninstall fastai` or `conda uninstall fastai`) and replace it with a [pip editable install](http://codumentary.blogspot.com/2014/11/python-tip-of-year-pip-install-editable.html):

    git clone https://github.com/fastai/fastai_pytorch
    cd fastai_pytorch
    pip install -e .
    tools/run-after-git-clone

Please refer to [CONTRIBUTING.md](https://github.com/fastai/fastai_pytorch/blob/master/CONTRIBUTING.md) and [the developers guide](http://docs.fast.ai/developers.html) for more details.
