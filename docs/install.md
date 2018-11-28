---
title: Installation
---

## Basic installation

Please refer to [README](https://github.com/fastai/fastai/blob/master/README.md#installation) for bulk of the instructions



## Can't install the latest conda fastai package

If you do:

```
conda install -c fastai fastai
```

and conda installs not the latest `fastai` version, but an older one, that means your conda environment has a conflict of dependencies with another previously installed package, that pinned one of its dependencies to a fixed version and only `fastai` older version's dependencies agree with that fixed version number. Unfortunately, conda is not user-friendly enough to tell you that. You may have to add the option `-v`, `-vv`, or `-vvv` after `conda install` and look through the verbose output to find out which package causes the conflict.

```
conda install -v -c fastai fastai
```

Here is a little example to understand the `conda` package dependency conflict:

Let's assume anaconda.org has 3 packages: `A`, `B` and `P`, and some of them have multiple release versions:

```
package A==1.0.17 depends on package P==1.0.5
package B==1.0.06 depends on package P==1.0.5
package B==1.0.29 depends on package P==1.0.6
package P==1.0.5
package P==1.0.6
```

If you installed `A==1.0.17` via conda, and are now trying to install `conda install B`, conda will install an older `B==1.0.6`, rather than the latest `B==1.0.29`, because the latter needs a higher version of `P`. conda can't install `B==1.0.29` because then it'll break the dependency requirements of the previously installed package `A`, which needs `P==1.0.5`. However, if conda installs `B==1.0.6` there is no conflict, as both `A==1.0.17` and `B==1.0.6` agree on dependency `P==1.0.5`.

It'd have been nice for `conda` to just tell us that: there is a newer package `B` available, but it can't install it because it conflicts on dependency with package `A` which needs package `P` of such and such version. But, alas, this is not the case.

The easiest solution to this problem is to create a new dedicated conda environment for fastai as the documentation suggests.

Alternatively, use `pip install` (onto the same conda env), but it'll probably break that other package that depends on some fixed number of another dependency.



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
