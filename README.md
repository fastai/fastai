# fastai

The fastai library simplifies training fast and accurate neural nets using modern best practices. See the [fastai website](http://docs.fast.ai) to get started. The library is based on research in to deep learning best practices undertaken at [fast.ai](http://www.fast.ai), and includes \"out of the box\" support for [`vision`](http://docs.fast.ai/vision.html#vision), [`text`](http://docs.fast.ai/text.html#text), [`tabular`](http://docs.fast.ai/tabular.html#tabular), and [`collab`](http://docs.fast.ai/collab.html#collab) (collaborative filtering) models. For brief examples, see the [examples](https://github.com/fastai/fastai/tree/master/examples) folder; detailed examples are provided in the full documentation. For instance, here's how to train an MNIST model using [resnet18](https://arxiv.org/abs/1512.03385) (from the [vision example](https://github.com/fastai/fastai/blob/master/examples/vision.ipynb)):

```python
untar_data(MNIST_PATH)
data = image_data_from_folder(MNIST_PATH)
learn = ConvLearner(data, tvm.resnet18, metrics=accuracy)
learn.fit(1)
```

## Note for [course.fast.ai](http://course.fast.ai) students

If you are using fastai for any [course.fast.ai](http://course.fast.ai) course, please do *NOT* install fastai from pip or conda using the instructions below; the instructions below are for fastai v1, but the courses use fastai 0.7. For the courses, you should simply follow the instructions in the course (i.e. clone this repo, cd to it, and `conda env update`), and the notebooks will work (there is a symlink to old/fastai/, which is fastai 0.7, in each course notebook directory), or else use `pip install fastai==0.7.0` to install the version compatible with the course.

*Note: If you want to learn how to use fastai v1 from its lead developer, Jeremy Howard, he will be teaching it in the [Deep Learning Part I](https://www.usfca.edu/data-institute/certificates/deep-learning-part-one) course at the University of San Francisco from Oct 22nd, 2018.*

## Installation

`fastai-1.x` can be installed with either `conda` or `pip` package managers and also from source. At the moment you can't just run *install*, since you first need to get the correct `pytorch` version installed - thus to get `fastai-1.x` installed choose one of the installation recipes below using your favourite python package manager.

### Conda Install

Follow the following 2 steps in this exact order:

1. Install the nightly `pytorch` build, with `cudaXX` package version matching your system's setup. For example, for CUDA 9.2:
   ```
   conda install -c pytorch pytorch-nightly cuda92
   ```
   If you have a different CUDA version, find the right build [here](https://pytorch.org/get-started/locally/). Choose Preview/Your OS/Conda/Python3.6|Python3.7 and your CUDA version and it will give you the correct install instruction. Instructions to build `pytorch` from source are provided at the same location.

   If your system doesn't have CUDA, you can install the CPU-only `pytorch` build:

   ```
   conda install -c pytorch pytorch-nightly-cpu
   ```

2. Install `fastai`:

   ```
   conda install -c fastai fastai
   ```

   NB: We are currently using a re-packaged `torchvision` as `torchvision-nightly` in order to support `pytorch-nightly`, which is required for using `fastai`.

If you encounter installation problems, make sure you have the latest `conda` client:
```
conda update conda
```

If the issue persists, please read about [installation issues](README.md#installation-issues).

### PyPI Install

Follow the following 2 steps in this exact order:

1. Install the nightly `pytorch` build, with `cudaXX` package version matching your system's setup. For example for CUDA 9.2:

   ```
   pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
   ```

   If you have a different CUDA version, find the right build [here](https://pytorch.org/get-started/locally/). Choose Preview/Your OS/Pip/Python3.6|Python3.7 and your CUDA version and it will give you the correct install instruction. Instructions to build `pytorch` from source are provided at the same location.

If your system doesn't have CUDA, you can install the CPU-only `torch` build:

   ```
   pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
   ```

2. Install `fastai`:

   ```
   pip install fastai
   ```

   NB: this will also fetch `torchvision-nightly`, see the conda entry above for details.

If you experience installation problems, please read about [installation issues](README.md#installation-issues).


### Developer Install

First, follow the instructions above for either `PyPi` or `Conda`. Then uninstall the `fastai` package using the **same package manager you used to install it**, i.e. `pip uninstall fastai` or `conda uninstall fastai`, and then, replace it with a [pip editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs).


```
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e .[dev]
```

You can test that the build works by starting the jupyter notebook:

```
jupyter notebook
```
and executing an example notebook. For example load `examples/tabular.ipynb` and run it.

Alternatively, you can do a quick CLI test:

```
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to notebook examples/tabular.ipynb
```

If anything goes wrong please [read and report installation
issues](http://forums.fast.ai/t/fastai-v1-install-issues-thread/24111).

Please refer to [CONTRIBUTING.md](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md) and  [develop.md](https://github.com/fastai/fastai/blob/master/docs-dev/develop.md) for more details on how to contribute to the `fastai` project.


## Installation Issues

If the installation process fails, first make sure [your system is supported](README.md#is-my-system-supported). And if the problem is still not addressed, please see  [this installation issues thread](http://forums.fast.ai/t/fastai-v1-install-issues-thread/24111).


### Is My System Supported?

1. Python: You need to have python 3.6 or higher

2. Operating System:

   Since fastai-1.0 relies on pytorch-1.0, you need to be able to install pytorch-1.0 first.

   As of this moment pytorch.org's pre-1.0.0 version (`torch-nightly`) supports:

    | Platform | GPU    | CPU    |
    | ---      | ---    | ---    |
    | linux    | binary | binary |
    | mac      | source | binary |
    | windows  | source | source |

   Legend: `binary` = can be installed directly, `source` = needs to be built from source.

   This will change once `pytorch` 1.0.0 is released and installable packages made available for your system, which could take some time after the official release is made. Please watch for updates [here](https://pytorch.org/get-started/locally/).

   If your system is currently not supported, please consider installing and using the very solid "v0" version of `fastai`. Please see the [instructions](https://github.com/fastai/fastai/tree/master/old).







## Copyright

Copyright 2017 onwards, fast.ai, Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
