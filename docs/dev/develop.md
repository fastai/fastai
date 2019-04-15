---
title: Notes For Developers
---

## Things to Run After git clone

Make sure you follow this recipe:

    git clone https://github.com/fastai/fastai
    cd fastai
    tools/run-after-git-clone

This will take care of everything that is explained in the following sections. That is `tools/run-after-git-clone` will execute the scripts that are explained individually below. You still need to know what they do, but you need to execute just one script.

Note: windows users, not using bash emulation, will need to invoke the command as:

    python tools\run-after-git-clone

This applies to the following `fastai` github user repos: `fastai`, `fastai_docs`, `course-v3`.

### after-git-clone #1: a mandatory notebook strip out

Currently we only store `source` code cells under git (and a few extra fields for documentation notebooks). If you would like to commit or submit a PR, you need to confirm to that standard.

This is done automatically during `diff`/`commit` git operations, but you need to configure your local repository once to activate that instrumentation.

Therefore, your developing process will always start with:

    tools/trust-origin-git-config

The last command tells git to invoke configuration stored in `fastai/.gitconfig`, so your `git diff` and `git commit` invocations for this particular repository will now go via `tools/fastai-nbstripout` which will do all the work for you.

You don't need to run it if you run:

    tools/run-after-git-clone

If you skip this configuration your commit/PR involving notebooks will not be accepted, since it'll carry in it many JSON bits which we don't want in the git repository. Those unwanted bits create collisions and lead to unnecessarily complicated and time wasting merge activities. So please do not skip this step.

Note: we can't make this happen automatically, since git will ignore a repository-stored `.gitconfig` for security reasons, unless a user will tell git to use it (and thus trust it).

If you'd like to check whether you already trusted git with using `fastai/.gitconfig` please look inside `fastai/.git/config`, which should have this entry:

    [include]
            path = ../.gitconfig

or alternatively run:

    tools/trust-origin-git-config -t



### after-git-clone #2: automatically updating doc notebooks to be trusted on git pull

We want the doc notebooks to be already trusted when you load them in `jupyter notebook`, so this script which should be run once upon `git clone`, will install a `git` `post-merge` hook into your local check out.

The installed hook will be executed by git automatically at the end of `git pull` only if it triggered an actual merge event and that the latter was successful.

To trust, run:

    tools/trust-doc-nbs-install-hook

You don't need to run it if you run:

    tools/run-after-git-clone

To distrust run:

    rm .git/hooks/post-merge



## Stripping Out Jupyter Notebooks

Our setup on all `fastai` projects requires that `*.ipynb` notebooks get stripped during the commit, which is accomplished by `fastai-nbstripout` which runs as a filter during `git commit`. Therefore, when you clone any of the `fastai` projects that contain jupyter notebooks you must always run:

```
tools/run-after-git-clone
```
which registers the filters. This needs to be done once per `git clone`.

Unfortunately, we can't enforce this, because github doesn't allow server-side hooks.

So it's your responsibility to watch the status of your commits at the commits page:

* [fastai](https://github.com/fastai/fastai/commits)
* [fastai_docs](https://github.com/fastai/fastai_docs/commits)
* [course-v3](https://github.com/fastai/course-v3/commits)

Alternatively, you can watch CI builds for the project you committed to:

* [fastai @ azure CI](https://dev.azure.com/fastdotai/fastai/_build?definitionId=7)
* [fastai_docs @ azure CI](https://dev.azure.com/fastdotai/fastai/_build?definitionId=11)
* [course-v3 @ azure CI](https://dev.azure.com/fastdotai/fastai/_build?definitionId=10)

It's very important that you do that on a consistent basis, because when you make this mistake you affect everybody who works on the same project. You basically make it impossible for other developers to `git pull` without some workarounds.


## Unstripped Notebook Repair

If you or someone forgot to run `tools/run-after-git-clone` after `git clone` and committed unstripped notebooks, here is how to repair it in the `fastai` repo:

1. disable the filter

   ```
   tools/trust-origin-git-config -d
   ```

2. strip out the notebooks

   ```
   tools/fastai-nbstripout -d docs_src/*ipynb courses/*/*ipynb examples/*ipynb
   ```

3. commit

   ```
   git commit path/to/notebooks
   git push
   ```

4. re-enable the filter (very important!)

   ```
   tools/trust-origin-git-config -e
   ```

Inside the `course-v3` repo, it'd be the same, but since the notebooks are in a different location, step 2 is:

   ```
   tools/fastai-nbstripout -d nbs/*/*ipynb
   ```

In the `fastai_docs` repo, we have two different types of notebooks: "code" and "docs" notebooks, therefore in step 2 we strip them out differently:

   ```
   tools/fastai-nbstripout dev_nb/*ipynb dev_nb/experiments/*ipynb
   tools/fastai-nbstripout -d dev_course/*/*ipynb dev_swift/*ipynb
   ```

Here are the quick copy-n-paste recipes (that assume you don't have anything else modified):

* Unix:

   The `fastai` repo:
   ```
   tools/trust-origin-git-config -d
   tools/fastai-nbstripout -d docs_src/*ipynb courses/*/*ipynb examples/*ipynb
   git commit docs_src courses examples
   git push
   tools/trust-origin-git-config -e
   ```

   The `fastai_docs` repo:
   ```
   tools/trust-origin-git-config -d
   tools/fastai-nbstripout dev_nb/*ipynb dev_nb/experiments/*ipynb
   tools/fastai-nbstripout -d dev_course/*/*ipynb dev_swift/*ipynb
   git commit dev_nb dev_course
   git push
   tools/trust-origin-git-config -e
   ```
   or just:
   `make strip`

   The `course-v3` repo:
   ```
   tools/trust-origin-git-config -d
   tools/fastai-nbstripout -d nbs/*/*ipynb
   git commit nbs
   git push
   tools/trust-origin-git-config -e
   ```

* Windows:

   The `fastai` repo:
   ```
   python tools\trust-origin-git-config -d
   python tools\fastai-nbstripout -d docs_src\*ipynb courses\*\*ipynb examples\*ipynb
   git commit docs_src courses examples
   git push
   python tools\trust-origin-git-config -e
   ```

   The `fastai_docs` repo:
   ```
   python tools\trust-origin-git-config -d
   python tools\fastai-nbstripout dev_nb\*ipynb dev_nb\experiments\*ipynb
   python tools\fastai-nbstripout -d dev_course\*\*ipynb dev_swift\*ipynb
   git commit dev_nb dev_course
   git push
   python tools\trust-origin-git-config -e
   ```

   The `course-v3` repo:

   ```
   python tools\trust-origin-git-config -d
   python tools\fastai-nbstripout -d nbs\*\*ipynb
   git commit nbs
   git push
   python tools\trust-origin-git-config -e
   ```


## Development Editable Install

For deploying the `fastai` repo's files, while being able to edit them, make sure to uninstall any previously installed `fastai`:

   ```
   pip   uninstall fastai
   conda uninstall fastai
   ```

And then do an editable install, from inside the cloned `fastai` directory:

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

If you're new to editable install, refer to [Editable installs](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
and its [examples](https://pip.pypa.io/en/stable/reference/pip_install/#examples).

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


## `fastai` Versions and Timeline

The timeline the `fastai` project follows is:

```
...
1.0.14
1.0.15.dev0
1.0.15
1.0.16.dev0
...
```

So that if your `fastai/version.py` or `fastai.__version__` doesn't include `.dev0` at the end, that means you're using a `fastai` release, which you installed via `pip` or `conda`. If you use a developer install as explained earlier, you will always have `.dev0` in the version number.

When a new release cycle starts it starts with `.dev0` in it, for example, `1.0.15.dev0`. When that cycle is complete and a release is made, it becomes `1.0.15`. Think of `.dev0` as a pre-release.


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




## Full Diffs Mailing List

If you'd like to follow closely the development of fastai, and you don't like clicking around github, we have a read-only full diffs mailing list that is open to all.

You can subscribe or unsubscribe at any time on your own accord [here](https://groups.google.com/forum/#!forum/fastai-diff).

If you need to comment on any diff you read on that list, click on the link on top of the email and it'll take you to the github page, corresponding to that commit, and you can comment there. Alternatively ask questions on the developer's section of the  [forums](https://forums.fast.ai/c/fastai-dev).

Make sure you use a decent email client, surely don't read the emails on google groups or gmail. You need to have a client that can properly render fixed fonts and not use variable fonts that look horrible with diffs. For example, [Thunderbird](https://www.thunderbird.net/) works well.

Chances are that your email client may put the emails into your spam folder, so make sure you tell your client they're ham!

You will probably want to filter these emails into a dedicated folder. If so, use the `List-ID` email header in the configuration of your email:

```
List-ID: <fastai-diff.googlegroups.com>
```

## Some useful oneliners

To fix links to have `.html` again (both needed):

```
perl -pi -e 's|href="(/[^"#]+)(#[^"]+)?"|href="$1.html$2"|g' docs/*html docs_src/*ipynb
perl -pi -e 's{https?://((?:docs|docs-dev|course-v3).fast.ai/)([\w\._-]+)(#[\w-_\.]+)?}{http://$1$2 .html$3}g' docs/*md
perl -pi -e 's{https?://((?:docs|docs-dev|course-v3).fast.ai/)}{https://$1}g' docs/*md README.md
```

How to safely and efficiently search/replace files in git repo using CLI. The operation must not touch anything under `.git`:
```
find . -type d -name ".git" -prune -o -type f -exec perl -pi -e 's|OLDSTR|NEWSTR|g' {} \;
```
but it `touch(1)`es all files which slows down git-side, so we want to do it on files that actually contain the old pattern:
```
grep --exclude-dir=.git -lIr "OLDSTR" . | xargs -n1 perl -pi -e 's|OLDSTR|NEWSTR|g'
```
