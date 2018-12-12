[![Build Status](https://dev.azure.com/fastdotai/fastai/_apis/build/status/fastai.fastai)](https://dev.azure.com/fastdotai/fastai/_build/latest?definitionId=1)
[![pypi fastai version](https://img.shields.io/pypi/v/fastai.svg)](https://pypi.python.org/pypi/fastai)
[![Conda fastai version](https://img.shields.io/conda/v/fastai/fastai.svg)](https://anaconda.org/fastai/fastai)

[![Anaconda-Server Badge](https://anaconda.org/fastai/fastai/badges/platforms.svg)](https://anaconda.org/fastai/fastai)
[![fastai python compatibility](https://img.shields.io/pypi/pyversions/fastai.svg)](https://pypi.python.org/pypi/fastai)
[![fastai license](https://img.shields.io/pypi/l/fastai.svg)](https://pypi.python.org/pypi/fastai)

# fastai

The fastai library simplifies training fast and accurate neural nets using modern best practices. See the [fastai website](https://docs.fast.ai) to get started. The library is based on research into deep learning best practices undertaken at [fast.ai](http://www.fast.ai), and includes \"out of the box\" support for [`vision`](https://docs.fast.ai/vision.html#vision), [`text`](https://docs.fast.ai/text.html#text), [`tabular`](https://docs.fast.ai/tabular.html#tabular), and [`collab`](https://docs.fast.ai/collab.html#collab) (collaborative filtering) models. For brief examples, see the [examples](https://github.com/fastai/fastai/tree/master/examples) folder; detailed examples are provided in the full [documentation](https://docs.fast.ai/). For instance, here's how to train an MNIST model using [resnet18](https://arxiv.org/abs/1512.03385) (from the [vision example](https://github.com/fastai/fastai/blob/master/examples/vision.ipynb)):

```python
untar_data(MNIST_PATH)
data = image_data_from_folder(MNIST_PATH)
learn = create_cnn(data, tvm.resnet18, metrics=accuracy)
learn.fit(1)
```

## Note for [course.fast.ai](http://course.fast.ai) students

If you are using `fastai` for any [course.fast.ai](http://course.fast.ai) course, you need to use `fastai 0.7`. Please ignore the rest of this document, which is written for `fastai v1`, and instead follow the installation instructions [here](https://forums.fast.ai/t/fastai-v0-install-issues-thread/24652).

*Note: If you want to learn how to use fastai v1 from its lead developer, Jeremy Howard, he will be teaching it in the [Deep Learning Part I](https://www.usfca.edu/data-institute/certificates/deep-learning-part-one) course at the University of San Francisco from Oct 22nd, 2018.*

## Installation

**NB:** *fastai v1 currently supports Linux only, and requires **PyTorch v1** and **Python 3.6** or later. Windows support is at an experimental stage: it should work fine but we haven't thoroughly tested it. Since Macs don't currently have good Nvidia GPU support, we do not currently prioritize Mac development.*

`fastai-1.x` can be installed with either `conda` or `pip` package managers and also from source. At the moment you can't just run *install*, since you first need to get the correct `pytorch` version installed - thus to get `fastai-1.x` installed choose one of the installation recipes below using your favourite python package manager. Note that **PyTorch v1** and **Python 3.6** are the minimal version requirements.

If your system has a [recent NVIDIA card](https://www.geforce.com/hardware/technology/cuda/supported-gpus) with the correctly configured NVIDIA driver please follow the GPU installation instructions. Otherwise, the CPU-ones.

It's highly recommended you install `fastai` and its dependencies in a virtual environment ([`conda`](https://conda.io/docs/user-guide/tasks/manage-environments.html) or others), so that you don't interfere with system-wide python packages. It's not that you must, but if you experience problems with any dependency packages, please consider using a fresh virtual environment just for `fastai`.

If you experience installation problems, please read about [installation issues](https://github.com/fastai/fastai/blob/master/README.md#installation-issues).

More advanced installation issues, such as installing only partial dependencies are covered in a dedicated [installation doc](https://docs.fast.ai/install.html).

### Conda Install

```bash
conda install -c pytorch -c fastai fastai
```

Note that JPEG decoding can be a bottleneck, particularly if you have a fast GPU. You can optionally install an optimized JPEG decoder as follows (Linux):

```bash
conda uninstall --force jpeg -y
conda install -c conda-forge libjpeg-turbo
CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd
```
For the full story see [Pillow-SIMD](https://docs.fast.ai/performance.html#installation).

### PyPI Install

```bash
pip install fastai
```

### Developer Install

First, follow the instructions above for either `PyPi` or `Conda`. Then uninstall the `fastai` package using the same package manager you used to install it, i.e. `pip uninstall fastai` or `conda uninstall fastai`, and then, replace it with a [pip editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs).


```bash
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e .[dev]
```

You can test that the build works by starting the jupyter notebook:

```bash
jupyter notebook
```
and executing an example notebook. For example load `examples/tabular.ipynb` and run it.

Alternatively, you can do a quick CLI test:

```bash
jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to notebook examples/tabular.ipynb
```

Please refer to [CONTRIBUTING.md](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md) and  [develop.md](https://github.com/fastai/fastai/blob/master/docs/develop.md) for more details on how to contribute to the `fastai` project.




### Building From Source

If for any reason you can't use the prepackaged packages and have to build from source, this section is for you.

1. To build `pytorch` from source follow the [complete instructions](https://github.com/pytorch/pytorch#from-source). Remember to first install CUDA, CuDNN, and other required libraries as suggested - everything will be very slow without those libraries built into `pytorch`.

2. Next, you will also need to build `torchvision` from source:

   ```bash
   git clone https://github.com/pytorch/vision
   cd vision
   python setup.py install
   ```

3. When both `pytorch` and `torchvision` are installed, first test that you can load each of these libraries:

   ```bash
   import torch
   import torchvision
   ```

   to validate that they were installed correctly

   Finally, proceed with `fastai` installation as normal, either through prepackaged pip or conda builds or installing from source ("the developer install") as explained in the sections above.



## Installation Issues

If the installation process fails, first make sure [your system is supported](https://github.com/fastai/fastai/blob/master/README.md#is-my-system-supported). And if the problem is still not addressed, please refer to the [troubleshooting document](https://docs.fast.ai/troubleshoot.html).

If you encounter installation problems with conda, make sure you have the latest `conda` client (`conda install` will do an update too):
```bash
conda install conda
```

### Is My System Supported?

1. Python: You need to have python 3.6 or higher

2. CPU or GPU

   The `pytorch` binary package comes with its own CUDA, CuDNN, NCCL, MKL, and other libraries so you don't have to install system-wide NVIDIA's CUDA and related libraries if you don't need them for something else. If you have them installed already it doesn't matter which NVIDIA's CUDA version library you have installed system-wide. Your system could have CUDA 9.0 libraries, and you can still use `pytorch` build with `cuda9.2` libraries without any problem, since the `pytorch` binary package is self-contained.

   The only requirement is that you have installed and configured the NVIDIA driver correctly. Usually you can test that by running `nvidia-smi`. While it's possible that this application is not available on your system, it's very likely that if it doesn't work, than your don't have your NVIDIA drivers configured properly. And remember that a reboot is always required after installing NVIDIA drivers.

3. Operating System:

   Since fastai-1.0 relies on pytorch-1.0, you need to be able to install pytorch-1.0 first.

   As of this moment pytorch.org's 1.0 version supports:

    | Platform | GPU    | CPU    |
    |----------|--------|--------|
    | linux    | binary | binary |
    | mac      | source | binary |
    | windows  | binary | binary |

   Legend: `binary` = can be installed directly, `source` = needs to be built from source.

   If there is no `pytorch` preview conda or pip package available for your system, you may still be able to [build it from source](https://pytorch.org/get-started/locally/).

4. How do you know which pytorch cuda version build to choose?

   It depends on the version of the installed NVIDIA driver. Here are the requirements for CUDA versions supported by pre-built `pytorch` releases:

    | CUDA Toolkit | NVIDIA (Linux x86_64) |
    |--------------|-----------------------|
    | CUDA 10.0    | >= 410.00             |
    | CUDA 9.0     | >= 384.81             |
    | CUDA 8.0     | >= 367.48             |

   So if your NVIDIA driver is less than 384, then you can only use `cuda80`. Of course, you can upgrade your drivers to more recent ones if your card supports it.

   You can find a complete table with all variations [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

   If you use NVIDIA driver 410+, you most likely want to install the `cuda100` pytorch variant, via:
   ```bash
   conda install -c pytorch pytorch cuda100
   ```
   or if you need a lower version (`cuda90` is installed by default), use:
   ```bash
   conda install -c pytorch pytorch cuda80
   ```
   For other options refer to the complete list of [the available pytorch variants](https://pytorch.org/get-started/locally/).

## Updates

In order to update your environment, simply install `fastai` in exactly the same way you did the initial installation.

Top level files `environment.yml` and `environment-cpu.yml` belong to the old fastai (0.7). `conda env update` is no longer the way to update your `fastai-1.x` environment. These files remain because the fastai course-v2 video instructions rely on this setup. Eventually, once fastai course-v3 p1 and p2 will be completed, they will probably be moved to where they belong - under `old/`.



## History

A detailed history of changes can be found [here](https://github.com/fastai/fastai/blob/master/CHANGES.md).



## Copyright

Copyright 2017 onwards, fast.ai, Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
