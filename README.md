[![CI-Badge](https://github.com/fastai/fastai2/workflows/CI/badge.svg)](https://github.com/fastai/fastai2/actions?query=workflow%3ACI) [![PyPI](https://img.shields.io/pypi/v/fastai2?color=blue&label=pypi%20version)](https://pypi.org/project/fastai2/#description) [![Conda (channel only)](https://img.shields.io/conda/vn/fastai/fastai2?color=seagreen&label=conda%20version)](https://anaconda.org/fastai/fastai2) [![Build fastai2 images](https://github.com/fastai/docker-containers/workflows/Build%20fastai2%20images/badge.svg)](https://github.com/fastai/docker-containers)

# Welcome to fastai v2
> NB: This is still in early development. Use v1 unless you want to contribute to the next version of fastai


To learn more about the library, read our introduction in the [paper](https://arxiv.org/abs/2002.04688) presenting it.

Note that the docs are in a submodule, so to clone with docs included, you should use:

     git clone --recurse-submodules https://github.com/fastai/fastai2
     
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Installing](#installing)
- [Tests](#tests)
- [Contributing](#contributing)
- [Docker Containers](#docker-containers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installing

You can get all the necessary dependencies by simply installing fastai v1: `conda install -c fastai -c pytorch fastai`. Or alternatively you can automatically install the dependencies into a new environment:

```bash
conda env create -f environment.yml
source activate fastai2
```

Then, you can install fastai v2 with pip: `pip install fastai2`. 

Or you can use an editable install (which is probably the best approach at the moment, since fastai v2 is under heavy development):
``` 
git clone --recurse-submodules https://github.com/fastai/fastai2
cd fastai2
pip install -e ".[dev]"
``` 
You should also use an editable install of [`fastcore`](https://github.com/fastai/fastcore) to go with it.

If you want to browse the notebooks and build the library from them you will need nbdev:
``` 
pip install nbdev
``` 

To use `fastai2.medical.imaging` you'll also need to:

```bash
conda install pyarrow
pip install pydicom kornia opencv-python scikit-image
```

## Tests

To run the tests in parallel, launch:

```bash
nbdev_test_nbs
```
or 
```bash
make test
```

For all the tests to pass, you'll need to install the following optional dependencies:

```
pip install "sentencepiece<0.1.90" wandb tensorboard albumentations pydicom opencv-python scikit-image pyarrow
pip install kornia --no-deps
```

## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The script `nbdev_diff_nbs` can let you know if there is a difference between the local library and the notebooks.
* If you made a change to the notebooks in one of the exported cells, you can export it to the library with `nbdev_build_lib` or `make fastai2`.
* If you made a change to the library, you can export it back to the notebooks with `nbdev_update_lib`.

## Docker Containers

For those interested in offical docker containers for this project, they can be found [here](https://github.com/fastai/docker-containers#fastai2).
