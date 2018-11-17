---
title: Installation
---

## Basic installation

Please refer to [README](https://github.com/fastai/fastai/blob/master/README.md#installation) for bulk of the instructions


## Custom Dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have a limited disc space on your remote instance, here how you can install only the dependencies that you need.

First, install `fastai` without its dependencies, and then install the dependencies that you need directly:

```
pip install --no-deps fastai
pip install "fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" ...
```
this will work with conda too:

```
conda install --no-deps fastai
conda install "fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" ...
```

Below you will find the groups of dependencies for you to choose from. `fastai.base` is mandatory, the rest are optional:

```
fastai.base:

"fastprogress>=0.1.15" "matplotlib" "numpy>=1.12" "pandas" "bottleneck" "numexpr" "Pillow" "requests" "scipy" "typing" "pyyaml"

fastai.text:

"spacy==2.0.16" "regex" "thinc==6.12.0" "cymem==2.0.2"

fastai.vision:

"torchvision-nightly"
```
