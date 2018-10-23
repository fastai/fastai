---
title: Troubleshooting
---

## Initial Installation


### Correctly configured NVIDIA drivers

Please skip this section if  your system doesn't have an NVIDIA GPU.

It doesn't matter which CUDA version you have installed on your system, always try first to install the latest `pytorch-nightly` with `cuda92` - it has all the required libraries built into the package. However, note, that you most likely will **need 396.xx+ driver for `pytorch` built with `cuda92`**. For older drivers you will probably need to install `pytorch` with `cuda90` or ever earlier.

The only thing you to need to ensure is that you have a correctly configured NVIDIA driver, which usually you can test by running: `nvidia-smi` in your console.

Please note that some users have **more than one version of NVIDIA driver installed** (e.g. one via `apt` and another from source). It's very likely that `pytorch` may have issues on such system. In such case, please, wipe any remnants of NVIDIA driver on your system, and install one NVIDIA driver of your choice.

Once you uninstalled the old drivers, to make sure you don't have any orphaned nvidia drivers on your system, usually it's enough to run:

`find /usr/ | grep libcuda.so`

or a more sweeping one (if your system updates the `mlocate` `db` regularly)

`locate -e libcuda.so`

If you get any listing as a result of  running of these commands, that means you still have some orphaned nvidia driver on your system.

Also note that `pytorch` will **silently fallback to CPU** if it reports `torch.cuda.is_available()` as `False`, so the only indicator of something being wrong will be that your notebooks will be running very slowly and you will hear your CPU revving up (if you are using a local system). Run:

```
python -c 'import fastai; fastai.show_install(1)'
```
to detect such issues. If you have this problem it'll say that your torch cuda is not available.



### Do not mix conda-forge packages

`fastai` depends on a few packages that have a complex dependency tree, and `fastai` has to manage those very carefully, so in the conda-land we rely on the anaconda main channel and test everything against that.

If you install some packages from the `conda-forge` channel you may have problems with `fastai` installation since you may have wrong versions of packages installed.

So see which packages need to come from the main conda channel do:

```
conda search --info -c fastai "fastai=1.0.7"
```

Replace `1.0.7` with the version you're trying to install. (Without version it'll show you all versions of `fastai` and its dependencies)



### Dedicated environment

`fastai` has a relatively complex set of python dependencies, and it's the best not to install those system-wide, but to use a virtual environment instead (`[conda](https://conda.io/docs/user-guide/tasks/manage-environments.html)` or others). A lot of problems disappear when a fresh dedicated to `fastai` virtual environment is created.

The following example is for using a conda environment.

First you need to install [miniconda](https://conda.io/docs/install/quick.html) or [anaconda](https://docs.anaconda.com/anaconda/install/). The former comes with bare minimum of packages preinstalled, the latter has hundreds more. If you haven't changed the default configuration, miniconda usually ends up under `~/miniconda3/`, and anaconda under `~/anaconda3/`.

Once you have the software installed, here is a quick way to set up a dedicated environment for just `fastai` with `python-3.6` (of course feel free to name it the way you want it to):

```
conda update conda
conda create -y python=3.6 --name fastai-3.6
conda activate fastai-3.6
conda install -y conda
conda install -y pip setuptools
```

Now you can [install `fastai` prerequisites and itself](https://github.com/fastai/fastai/blob/master/README.md#conda-install) using `conda`.

The only thing you need to remember when you start using a virtual environment is that you must activate it before using it. So for example when you open a new console and want to start `jupyter`, instead of doing:


```
jupyter notebook
```

you'd change your script to:

```
conda activate fastai-3.6
jupyter notebook
```

sometimes when you're outside of conda the above doesn't work and you need to do:

```
source ~/anaconda3/bin/activate fastai-3.6
jupyter notebook
```

(of course adjust the path to your conda installation if need to).

Virtual environments provide a lot of conveniences - for example if you want to have a stable env and an experimental one you can clone them in one command:

```
conda create --name fastai-3.6-experimental --clone fastai-3.6
```

or say you want to see how the well-working python-3.6 env will work with python-3.7:

```
conda create --name fastai-3.7 --clone fastai-3.6
conda install -n fastai-3.7 python=3.7
conda update -n fastai-3.7 --all
```

If you use advanced bash prompt functionality, like with [git-prompt](https://github.com/magicmonty/bash-git-prompt), it'll now tell you automatically which environment has been activated, no matter where you're on your system. e.g. on my setup it shows:

```
             /fastai:[master|✚1…4⚑3] > conda activate
(base)       /fastai:[master|✚1…4⚑3] > conda activate fastai-3.6
(fastai-3.6) /fastai:[master|✚1…4⚑3] > conda deactivate
             /fastai:[master|✚1…4⚑3] >
```

I tweaked the prompt output for this example by adding whitespace to align the entries to make it easy to see the differences. That leading white space is not there normally. Besides the virtual env, it also shows me which git branch I'm on, and various git status information.

So now you don't need to guess and you know exactly which environment has been activated if any before you execute any code.

### Am I using my GPU(s)?

It's possible that your system is misconfigured and while you think you're using your GPU you could be running on your CPU only.

You can check that by checking the output of `torch.cuda.is_available()` - it should return `True` if `pytorch` sees your GPU(s). You can also see the state of your setup with:

```
python -c 'import fastai; fastai.show_install(1)'
```
which will include that check in its report.

But the simplest direct check is to observe the output of `nvidia-smi` while you run your code. If you don't see the process show up when you run the code, then you aren't using your GPU.

To do that you can poll `nvidia-smi`s output with either:

```
watch -n 1 nvidia-smi
```

or alternatively with:

```
nvidia-smi dmon
```

The former is useful for watching the processes, the latter provides an easier way to see usage stats as their change.

If you're on a local system, another way to tell you're not using your GPU would be an increased noise from your CPU fan.


## Installation Updates

Please skip this section unless you have post- successful install update issues.

Normally you'd update `fastai` by running `pip install -U fastai` or `conda update fastai`, using the same package manager you used to install it in first place (but in reality, either will work).

If you use the [developer setup](https://github.com/fastai/fastai/blob/master/README.md#developer-install), then you need to do a simple:

```
cd path/to/your/fastai/clone
git pull
```

Sometimes jupyter notebooks get messed up, and `git pull` might fail with an error like:
```
error: Your local changes to the following files would be overwritten by merge:
examples/cifar.ipynb
Please, commit your changes or stash them before you can merge.
Aborting
```
then either make a new `fastai` clone (the simplest), or resolve it: disable the nbstripout filter, clean up your checkout and re-enable the filter.

    tools/trust-origin-git-config -d
    git stash
    git pull
    tools/trust-origin-git-config

Of course `git stash pop` if you made some local changes and you want them back.

You can also overwrite the folder if you have no changes you want to keep with `git checkout examples` in this case, or do a reset - but you have to do it **after** you disabled the filters, and then remember to **re-enable** them back.  The instructions above do it in a non-destructive way.

The stripout filter allows us to collaborate on the notebooks w/o having conflicts with different execution counts, locally installed extensions, etc., keeping under git only the essentials. Ideally, even the `outputs` should be stripped, but that's a problem if one uses the notebooks for demo, as it is the case with `examples` notebooks.



## Managing Multiple Installations

It's possible to have multiple `fastai` installs - usually in different conda environments. And when you do that it's easy to get lost in which environment of `fastai` you currently use.

Other than `conda activate wanted-env` here are some tips to find your way around:

Tell me which environment modules are imported from:
```
import sys
print(sys.path)
```

Tell me which `fastai` library got loaded (we want to know the exact location)
```
import sys, fastai
print(sys.modules['fastai'])
```

At times a quick hack can be used to get your first notebook working and then sorting out the setup. Say you checked out `fastai` to `/tmp/`:
```
cd /tmp/
git clone https://github.com/fastai/fastai
cd fastai
```
So now you know that your *uninstalled* `fastai` is located under `/tmp/fastai/`. Next, put the following on the very top of your notebook:
```
import sys
sys.path.append("/tmp/fastai")
import fastai
```
and it should just work. Now, go and sort out the rest of the installation, so that you don't need to do it for every notebook.



## Conda environments not showing up in Jupyter Notebook

While normally you shouldn't have this problem, and all the required things should get installed automatically, some users report that their jupyter notebook
does not recognize newly created environments at times. They reported the following to work:

```
conda activate fastai-3.6
conda install jupyter
conda install nb_conda
conda install nb_conda_kernels
conda install ipykernel
python -m ipykernel install --user --name fastai-3.6 --display-name "Python (fastai-3.6)"
```
See also [Kernels for different environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments).


## Support

Before making a new issue report, please:

1.  Make sure you have the latest `conda` and/or `pip`, depending on the package manager you use:
    ```
    pip install pip -U
    conda update conda
    ```
    and then check whether the problem you wanted to report still exists.

2.  Make sure [your platform is supported by the preview build of `pytorch-1.0.0`](https://github.com/fastai/fastai/blob/master/README.md#is-my-system-supported). You may have to build `pytorch` from source if it isn't.

3. Make sure you follow [the exact installation instructions](https://github.com/fastai/fastai/blob/master/README.md). If you improvise and it works that's great, if it fails please RTFM ;)

If you followed the steps in this document and couldn't find a resolution, please post a comment in this [thread](https://forums.fast.ai/t/fastai-v1-install-issues-thread/24111/1).


If the issue is still relevant, make sure to include in your post:

1. the output of the following script (including the \`\`\`text opening and closing \`\`\` so that it's formatted properly in your post):
   ```
   git clone https://github.com/fastai/fastai
   cd fastai
   python -c 'import fastai; fastai.show_install(1)'
   ```

   If you already have a `fastai` checkout, then just update it first:
   ```
   cd fastai
   git pull
   python -c 'import fastai; fastai.show_install(1)'
   ```

   The reporting script won't work if `pytorch` wasn't installed, so if that's the case, then send in the following details:
   * output of `python --version`
   * your OS: linux/osx/windows / and linux distro+version if relevant
   * output of `nvidia-smi`  (or say CPU if none)

2. a brief summary of the problem
3. the exact installation steps you followed

If the resulting output is very long, please paste it to https://pastebin.com/ and include a link to your paste

### Do's and Don'ts:

* please do not send screenshots with trace/error messages - we can't copy-n-paste from the images, instead paste them verbatim into your post and use the markdown gui menu so that it's code-formatted.

* If your system is configured to use a non-English locale, if possible, re-run the problematic code after running:

   `export LC_ALL=en_US.UTF-8`

    So that the error messages will be in English. You can run `locale` to see which locales you have installed.

### Bug Reports and PRs

If you found a bug and know how to fix it please submit a PR with the fix [here](https://github.com/fastai/fastai/pulls). Thank you.
