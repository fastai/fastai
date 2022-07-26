---
title: Notes For Developers
---

## Development Editable Install

To do an editable install, from inside the cloned `fastai` directory:

   ```
   cd fastai
   pip install -e ".[dev]"
   ```

It's almost the same as:

   ```
   pip install -e .
   ```

but adding `[dev]` tells `pip` to install optional packages in the `dev` group of the `dev_requirements` dictionary variable in `fastai/setup.py`. These extra dependencies are needed only by developers and contributors.

It's best not to use `python setup.py develop` method [doc](https://setuptools.readthedocs.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode).

When you'd like to sync your codebase with the `master`, simply go back into the cloned `fastai` directory and update it:

   ```
   git pull
   ```

You don't need to do anything else.

### Editable Install Explained

If you're new to editable install, refer to [Editable installs](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs)
and its [examples](https://pip.pypa.io/en/stable/cli/pip_install/#pip-install-examples).

This section will demonstrate how the editable installs works with `fastai`, including some important nuances that are important to understand.

First, make sure you're in the correct [python environment](https://docs.fast.ai/install.html#virtual-environment) (`conda activate fastai`, or whatever you called your environment if any, perhaps you're using a system-wide install, then you don't need to activate anything, though it's much safer to use a dedicated virtual env for working with `fastai`).

Let's start by uninstalling `fastai`:
```
pip   uninstall -y fastai
conda uninstall -y fastai
```

`sys.path` is a list of system paths that python uses to search for modules to load during `import`.

Before an editable `fastai` install is added, we have the following `sys.path`:
```
python -c 'import sys, pprint; pprint.pprint(sys.path)'
['',
 '~/.local/lib/python3.6/site-packages',
 '~/anaconda3/envs/fastai/lib/python3.6/site-packages']
```
Several entries were removed to make the lists easier to compare.

Now let's perform an editable install for `fastai`:
```
cd ~/github
git clone https://github.com/fastai/fastai
cd fastai
pip install -e ".[dev]"
```

And let's look at  `sys.path` again:
```
python -c 'import sys, pprint; pprint.pprint(sys.path)'
['',
 '~/.local/lib/python3.6/site-packages',
 '~/anaconda3/envs/fastai/lib/python3.6/site-packages',
 '~/github/fastai']
```

You can see that the path of my github checkout of `fastai` was added to the end of the paths that python will search when it encounters `import fastai`.

This setup makes it possible to edit python modules under  `~/github/fastai/fastai/*/*py` and have python load those files in programs running inside `conda`'s `fastai` environment automatically.

And you can see how python+pip accomplish that:
```
pip uninstall fastai
Uninstalling fastai-1.0.38.dev0:
  Would remove: ~/anaconda3/envs/fastai/lib/python3.6/site-packages/fastai.egg-link
```
And inside `fastai.egg-link` you will find `~/github/fastai`.

One important lesson here is that you must not have a normally installed `fastai` co-exist with an editable install. As you can tell from the contents of `sys.path` the editable path is added last to the module search path. Therefore, if you have a normally installed `fastai` package, python will use that instead of the editable install, which is probably not what you want.

This problem doesn't exist with pip. If you install a pip `fastai` package and then follow with a pip editable install of the same, pip will uninstall the previously installed package, before performing the editable install. It will do the same if you try to install a pip `fastai` package over a pip editable install of the same. It does it right, ensuring there is only one version of it installed.

This is not the situation with conda packages. If you do:

```
conda install -c fastai fastai
cd ~/github/fastai
pip install -e ".[dev]"
```

You end up with 2 installations of `fastai`, having the conda `fastai` package loaded by python and the editable install practically invisible to python (as it will find the conda package first):

```
$ ls -l ~/anaconda3/envs/fastai/lib/python3.6/site-packages/fastai*
~/anaconda3/envs/fastai/lib/python3.6/site-packages/fastai
~/anaconda3/envs/fastai/lib/python3.6/site-packages/fastai-1.0.37-py3.7.egg-info
```

So if you script your editable installation, always make sure to uninstall any previously installed conda `fastai` packages:

```
pip   uninstall -y fastai
conda uninstall -y fastai
cd ~/github/fastai
pip install -e ".[dev]"
```

Also, note, that `conda` can also perform an editable install:

```
cd ~/github/fastai
conda develop .
python -c 'import sys, pprint; pprint.pprint(sys.path)'
['',
 '~/.local/lib/python3.6/site-packages',
 '~/anaconda3/envs/fastai/lib/python3.6/site-packages',
 '~/github/fastai']
```

It does exactly the same as pip, except it performs it by editing `~/anaconda3/envs/fastai/lib/python3.6/site-packages/conda.pth`.

So, you may think that it's better to use this approach if conda is your preferred way.

We don't recommend using this approach, because, it doesn't play well with conda's normal installs (a normal conda package install will supersede the editable install at run time). Unlike pip, conda's normal packages are oblivious of their editable versions and vice versa - so you end up with having both and only one working. Moreover, conda doesn't support extra dependencies implemented by pip (`dev` dependencies).

To uninstall the editable conda version you must use:

```
cd ~/github/fastai
conda develop -u .
```

## Switching Conda Environments in Jupyter

Other than the normal switching environments with restarts:

   ```
   source activate env1
   jupyter notebook
   (Ctrl-C to kill jupyter)
   source activate env2
   jupyter notebook
   ```

You can install [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels), which provides a separate jupyter kernel for each conda environment, along with the appropriate code to handle their setup. This makes switching conda environments as simple as switching jupyter kernel (e.g. from the kernel menu). And you don't need to worry which environment you started `jupyter notebook` from - just choose the right environment from the notebook.

## Some useful oneliners

How to safely and efficiently search/replace files in git repo using CLI. The operation must not touch anything under `.git`:
```
find . -type d -name ".git" -prune -o -type f -exec perl -pi -e 's|OLDSTR|NEWSTR|g' {} \;
```
but it `touch(1)`es all files which slows down git-side, so we want to do it on files that actually contain the old pattern:
```
grep --exclude-dir=.git -lIr "OLDSTR" . | xargs -n1 perl -pi -e 's|OLDSTR|NEWSTR|g'
```

