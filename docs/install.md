---
title: Installation
---

## Basic installation

Please refer to [README](https://github.com/fastai/fastai/blob/master/README.md#installation) for bulk of the instructions

## CPU build

Generally, pytorch GPU build should work fine on machines that don't have a CUDA-capable GPU, and will just use the CPU. However, you can install CPU-only versions of Pytorch if needed with `fastai`.

* pip

   The pip ways is very easy:

   ```bash
   pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
   pip install fastai
   ```

   Just make sure to pick the correct torch wheel url, according to the needed platform, python and CUDA version, which you will find [here](https://pytorch.org/get-started/locally/).

* conda

   The conda way is more involved. Since we have only a single fastai package that relies on the default `pytorch` package working with and without GPU environment, if you want to install something custom you will have to manually tweak the dependencies. This is explained in detail [here](/install.html#custom-dependencies). So follow the instructions there, but replace `pytorch` with `pytorch-cpu`, and `torchvision` with `torchvision-cpu`.

Also, please note, that if you have an old GPU and `pytorch` fails because it can't support it, you can still use the normal (GPU) `pytorch` build, by setting the env var `CUDA_VISIBLE_DEVICES=""`, in which case pytorch will not try to check if you even have a GPU.


## Jupyter notebook dependencies

The `fastai` library doesn't require the jupyter environment to work, therefore those dependencies aren't included. So if you are planning on using `fastai` in the jupyter notebook environment, e.g. to run the `fastai` course lessons and you haven't already setup the jupyter environment, here is how you can do it.


* conda

   ```bash
   conda install jupyter notebook
   conda install -c conda-forge jupyter_contrib_nbextensions
   ```

   Some users also [seem to need](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook) this conda package to be able to choose the right kernel environment, however, most likely you won't need this package.

   ```bash
   conda install nb_conda
   ```

* pip

   ```bash
   pip install jupyter notebook jupyter_contrib_nbextensions
   ```


## Custom dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have limited disk space on your remote instance, here is how you can install only the dependencies that you need.

1. First, install `fastai` without its dependencies using either `pip` or `conda`:

   ```
   # pip
   pip install --no-deps fastai
   # conda
   conda install --no-deps -c fastai fastai
   ```

2. The rest of this section assumes you're inside the `fastai` git repo, since that's where `setup.py` resides. If you don't have the repository checked out, do:

   ```
   git clone https://github.com/fastai/fastai
   cd fastai
   tools/run-after-git-clone
   ```

3. Next, find out which groups of dependencies you want:

   ```
   python setup.py -q deps
   ```
   You should get something like:
   ```
   Available dependency groups: core, text, vision
   ```

   You need to use at least the `core` group.

   Do note that the `deps` command is a custom `distutils` extension, i.e. it only works in the `fastai` setup.

4. Finally, install the custom dependencies for the desired groups.

   For the sake of this demonstration, let's say you want to get the core dependencies (`core`), plus dependencies specific to computer vision (`vision`). The following command will give you the up-to-date dependencies for these two groups:

   ```
   python setup.py -q deps --dep-groups=core,vision
   ```
   It will return something like:
   ```
   Pillow beautifulsoup4 bottleneck dataclasses;python_version<'3.7' fastprogress>=0.1.18 matplotlib numexpr numpy>=1.12 nvidia-ml-py3 packaging pandas pyyaml requests scipy torch>=1.0.0 torchvision typing
   ```
   which can be fed directly to `pip install`:

   ```
   pip install $(python setup.py -q deps --dep-groups=core,vision)
   ```

   Since conda uses a slightly different syntax/package names, to get the same output suitable for conda, add `--dep-conda`:

   ```
   python setup.py -q deps --dep-groups=core,vision --dep-conda
   ```

   If your shell doesn't support `$()` syntax, it most likely will support backticks, which are deprecated in modern `bash`. (The two are equivalent, but `$()` has a superior flexibility). If that's your situation, use the following syntax instead:

   ```
   pip install `python setup.py -q deps --dep-groups=core,vision`
   ```

* Manual copy-n-paste case:

   If, instead of feeding the output directly to `pip` or `conda`, you want to do it manually via copy-n-paste, you need to quote the arguments, in which case add the `--dep-quote` option, which will do it for you:

   ```
   # pip:
   python setup.py -q deps --dep-groups=core,vision --dep-quote
   # conda:
   python setup.py -q deps --dep-groups=core,vision --dep-quote --dep-conda
   ```

   So the output for pip will look like:
   ```
   "Pillow" "beautifulsoup4" "bottleneck" "dataclasses;python_version<'3.7'" "fastprogress>=0.1.18" "matplotlib" "numexpr" "numpy>=1.12" "nvidia-ml-py3" "packaging" "pandas" "pyyaml" "requests" "scipy" "torch>=1.0.0" "torchvision" "typing"
   ```

* Summary:

   pip selective dependency installation:
   ```
   pip install --no-deps fastai
   pip install $(python setup.py -q deps --dep-groups=core,vision)
   ```

   same for conda:
   ```
   conda install --no-deps -c fastai fastai
   conda install -c pytorch -c fastai $(python setup.py -q deps --dep-conda --dep-groups=core,vision)
   ```

   adjust the `--dep-groups` argument to match your needs.


* Full usage:

   ```
   # show available dependency groups:
   python setup.py -q deps

   # print dependency list for specified groups
   python setup.py -q deps --dep-groups=core,vision

   # see all options:
   python setup.py -q deps --help
   ```


## Development dependencies

As explained in [Development Editable Install](/dev/develop.html#development-editable-install), if you want to work on contributing to fastai you will also need to install the optional development dependencies. In addition to the ways explained in the aforementioned document, you can also install `fastai` with developer dependencies without needing to check out the `fastai` repo.

* To install the latest released version of `fastai` with developer dependencies, do:

   `pip install "fastai[dev]"`

* To accomplish the same for the cutting edge master git version:

   `pip install "git+https://github.com/fastai/fastai#egg=fastai[dev]"`




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


Also see [bash-git-prompt](/dev/git.html#bash-git-prompt) which will help you tell at any moment which environment you're in.
