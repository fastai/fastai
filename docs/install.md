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


## Custom Dependencies

If for any reason you don't want to install all of `fastai`'s dependencies, since, perhaps, you have a limited disk space on your remote instance, here is how you can install only the dependencies that you need.

First, install `fastai` without its dependencies, and then install the dependencies that you need directly:

```
pip install --no-deps fastai
pip install "fastprogress>=0.1.18" "matplotlib" "numpy>=1.12" ...
```
this will work with conda too:

```
conda install --no-deps -c fastai fastai
conda install -c fastai "fastprogress>=0.1.18" "matplotlib" "numpy>=1.12" ...
```

Don't forget to add `-c fastai` for the conda installs, e.g. it's needed for `torchvision`.

Below you will find the groups of dependencies for you to choose from. `fastai.base` is mandatory, the rest are optional:

```
fastai.base:

  "fastprogress>=0.1.18" "matplotlib" "numpy>=1.12" "pandas" "bottleneck" "numexpr" "Pillow" "requests" "scipy" "typing" "pyyaml"

fastai.text:

  "spacy" "regex" "thinc" "cymem"

fastai.text.qrnn:

  "cupy"

fastai.vision:

  "torchvision"

```
