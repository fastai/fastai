---
title: Installation
---

## Basic installation

Please refer to [README](https://github.com/fastai/fastai/blob/master/README.md#installation) for bulk of the instructions

## CPU build

Generally pytorch GPU build should work fine on machines that don't have a CUDA-capable GPU, and will just use the CPU. However, you can install CPU-only versions of Pytorch if needed:

* conda

   ```bash
   conda install -c pytorch pytorch-cpu torchvision
   conda install -c fastai fastai
   ```

* pip

   ```bash
   pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
   pip install fastai
   ```


## Custom dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have a limited disk space on your remote instance, here is how you can install only the dependencies that you need.

First, install `fastai` without its dependencies, and then install the dependencies that you need directly:

```
pip install --no-deps fastai
pip install "matplotlib" "numpy>=1.12" "pandas" ...
```
this will work with conda too:

```
conda install --no-deps -c fastai fastai
conda install -c pytorch -c fastai "matplotlib" "numpy>=1.12" "pandas"  ...
```

Don't forget to add `-c fastai` for the conda installs, e.g. it's needed for `torchvision`.

Below you will find the groups of dependencies for you to choose from. `fastai.base` is mandatory, the rest are optional:

```
fastai.base:

   "matplotlib" "numpy>=1.12" "pandas" "fastprogress>=0.1.18" "bottleneck" "bs4" "numexpr" "Pillow" "requests" "scipy" "typing" "pyyaml" "pytorch" "packaging" "nvidia-ml-py3"

fastai.text:

  "spacy" "regex" "thinc" "cymem"

fastai.text.qrnn:

  "cupy"

fastai.vision:

  "torchvision"

```


## Virtual environment

It's highly recommended to use a virtual python environment for the `fastai` project, first because you could experiment with different versions of it (e.g. stable-release vs. bleeding edge git version), but also because it's usually a bad idea to install various python package into the system-wide python, because it's so easy to break the system, if it relies on python and its 3rd party packages for its functionality.

There are several implementations of python virtual environment, and the one we recommend is `conda` (anaconda), because we release our packages for this environment and pypi, as well. `conda` doesn't have all python packages available, so when that's the case we use `pip` to install whatever is missing.

You will find the instructions for installing conda on each platform [here](https://docs.anaconda.com/anaconda/install/). Once you followed the instructions and installed anaconda, you're ready to build you first environment. For the sake of this example we will use an environment name `fastai`, but you can name it whatever you'd like it to be.

The following will create a `fastai` env with python-3.6:

```
conda create -n fastai python=3.6
```

Now any time you'd like to work in this environment, just execute:

```
conda activate fastai
```

It's very important that you activate your environment before you start the jupyter notebook if you're using `fastai` notebooks.

Say, you'd like to have another env to test fastai with python-3.7, then you'd create another one with:

```
conda create -n fastai-py37 python=3.7
```

and to activate that one, you'd call:


```
conda activate fastai-py37
```

If you'd like to exit the environment, do:

```
conda deactivate
```

To list out the available environments
```
conda env list
```


Also see [bash-git-prompt](https://docs.fast.ai/dev/git.html#bash-git-prompt) which will help you tell at any moment which environment you're in.
