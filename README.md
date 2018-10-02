# fastai

The fastai library simplifies training fast and accurate neural nets using modern best practices. See the [fastai website](http://docs.fast.ai) to get started. The library is based on research in to deep learning best practices undertaken at [fast.ai](http://www.fast.ai), and includes \"out of the box\" support for [`vision`](http://docs.fast.ai/vision.html#vision), [`text`](http://docs.fast.ai/text.html#text), [`tabular`](http://docs.fast.ai/tabular.html#tabular), and [`collab`](http://docs.fast.ai/collab.html#collab) (collaborative filtering) models. For brief examples, see the [examples](https://github.com/fastai/fastai/tree/master/examples) folder; detailed examples are provided in the full documentation. For instance, here's how to train an MNIST model using [resnet18](https://arxiv.org/abs/1512.03385) (from the [vision example](https://github.com/fastai/fastai/blob/master/examples/vision.ipynb)):

```python
untar_data(MNIST_PATH)
data = image_data_from_folder(MNIST_PATH)
learn = ConvLearner(data, tvm.resnet18, metrics=accuracy)
learn.fit(1)
```

## Note for [course.fast.ai](http://course.fast.ai) students

If you are using fastai for any course.fast.ai course, please do *NOT* install fastai from pip or conda using the instructions below; the instructions below are for fastai v1, but the courses use fastai 0.7. For the courses, you should simply follow the instructions in the course (i.e. clone this repo, cd to it, and `conda env update`), and the notebooks will work (there is a symlink to old/fastai/, which is fastai 0.7, in each course notebook directory).

## Conda Install

To install fastai with pytorch-nightly + CUDA 9.2 simply run:

```
conda install -c pytorch -c fastai fastai pytorch-nightly cuda92
```

If your setup doesn't have CUDA support remove the `cuda92` above (in which case you'll only be able to train on CPU, not GPU, which will be much slower). For different versions of the CUDA toolkit, you'll need to install the appropriate CUDA conda package based on what you've got installed on your system (i.e. instead of `cuda92` in the above, pick the appropriate option for whichever toolkit version you have installed; to see a list of options type: `conda search "cuda*" -c pytorch`).

NB: We are currently using a re-packaged torchvision in order to support pytorch-nightly, which is required for using fastai.

If your system doesn't have CUDA, you can install the CPU-only torch build:

```
conda install -c pytorch -c fastai fastai pytorch-nightly==1.0.0.dev20180928=py3.6_cpu_0
```


## PyPI Install

First install the nightly `pytorch` build, e.g. for CUDA 9.2:

```
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
```

If you have a different CUDA version find the right build [here](https://pytorch.org/get-started/locally/). Choose Preview/Linux/Pip/python3.6|python3.7 and your CUDA version and it will give you the correct install instruction.

Next, install a custom `torchvision` build, that is built against `torch_nightly`.

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torchvision==0.2.1.post1
```

Now you can install `fastai`. Note, that this is a beta test version at the moment, please [report any issues](https://github.com/fastai/fastai/issues/):

```
pip install fastai
```

Sometimes, the last `pip` command still tries to get `torch-0.4.1`. If that happens to you, do:

```
pip uninstall torchvision fastai
pip install --no-deps torchvision
pip install fastai
```

## Is My System Supported?

1. Python: You need to have python 3.6 or higher

2. Operating System:

   Since fastai-1.0 relies on pytorch-1.0, you need to be able to install pytorch-1.0 first.

   As of this moment pytorch.org's pre-1.0.0 version (`torch-nightly`) supports:

     - linux: fully
     - mac: CPU-only
     - windows: not supported

   This will change once `pytorch` 1.0.0 is released and installable packages made available for your system, which could take some time after the official release is made. Please watch for updates [here](https://pytorch.org/get-started/locally/).

   If your system is currently not supported, please consider installing and using the very solid "v0" version of `fastai`. Please see the [instructions](https://github.com/fastai/fastai/tree/master/old).


## Developer Install

First, follow the instructions above for either `PyPi` or `Conda`. Then remove the fastai package (`pip uninstall fastai` or `conda uninstall fastai`) and replace it with a [pip editable install](http://codumentary.blogspot.com/2014/11/python-tip-of-year-pip-install-editable.html):

```
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e .
```

You can test that the build works:

```
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to notebook examples/tabular.ipynb
```

Please refer to [CONTRIBUTING.md](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md) and [the developers guide](http://docs.fast.ai/developers.html) for more details.

## Copyright

Copyright 2017 onwards, fast.ai, Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
