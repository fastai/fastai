---
title: Installation
---

## Basic installation

Please refer to [README](https://github.com/fastai/fastai/blob/master/README.md#installation) for bulk of the instructions


## Custom Dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have a limited disk space on your remote instance, here is how you can install only the dependencies that you need.

First, install `fastai` without its dependencies, and then install the dependencies that you need directly:

```
pip install --no-deps fastai
pip install "fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" ...
```
this will work with conda too:

```
conda install --no-deps -c fastai fastai
conda install -c fastai "fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" ...
```

Don't forget to add `-c fastai` for the conda installs, e.g. it's needed for `torchvision-nightly`.

Below you will find the groups of dependencies for you to choose from. `fastai.base` is mandatory, the rest are optional:

```
fastai.base:

  "fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" "pandas" "bottleneck" "numexpr" "Pillow" "requests" "scipy" "typing" "pyyaml"

fastai.text:

  "spacy==2.0.16" "regex" "thinc==6.12.0" "cymem==2.0.2"

fastai.text.qrnn:

  "cupy"

fastai.vision:

  "torchvision-nightly"
```
