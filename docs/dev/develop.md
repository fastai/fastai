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




## Development Editable Install

For deploying the `fastai` module's files, while being able to edit them, make sure to uninstall any previously installed `fastai`:

   ```
   pip   uninstall fastai
   conda uninstall fastai
   ```

And then do an editable install, from inside the cloned `fastai` directory:

   ```
   cd fastai
   pip install -e .[dev]
   ```

It's almost the same as:

   ```
   pip install -e .
   ```

but the former will also install extra dependencies needed only by developers.

Best not to use `python setup.py develop` method [doc](https://setuptools.readthedocs.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode).

When you'd like to sync your codebase with the `master`, simply go back into the cloned `fastai` directory and update it:


   ```
   git pull
   ```
You don't need to do anything else.

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



## Stripping Out Jupyter Notebooks

Our setup on all `fastai` projects requires that `*.ipynb` notebooks get stripped during the commit, which is accomplished by `fastai-nbstripout` which runs as a filter during `git commit`. Therefore, when you clone any of the `fastai` projects that contain jupyter notebooks you must always run:

```
tools/run-after-git-clone
```
which registers the filters. This needs to be done once per `git clone`.

Unfortunately, we can't enforce this, because github doesn't allow server-side hooks.

So it's your responsibility to watch the status of your commits at the commits page:

* https://github.com/fastai/fastai/commits/master
* https://github.com/fastai/course-v3/commits

Alternatively, you can watch CI builds for the project you committed to:

* https://dev.azure.com/fastdotai/fastai/_build?definitionId=7

It's very important that you do that on a consistent basis, because when you make this mistake you affect everybody who works on the same project. You basically make it impossible for other developers to `git pull` without some workarounds.

Should you make the mistake and commit some unstripped out notebooks, here is how you fix it:

1. disable the filter

   ```
   tools/trust-origin-git-config -d
   ```

2. strip out the notebook

   ```
   tools/fastai-nbstripout -d path/to/notebooks
   ```
   with an exception of `fastai_docs/dev_nb/*ipynb` notebooks, which need to be stripped with:
   ```
   tools/fastai-nbstripout path/to/notebooks
   ```
   without any arguments `outputs` are stripped, `-d` doesn't strip out the `outputs`.

3. commit

   ```
   git commit path/to/notebooks
   git push
   ```

4. re-enable the filter (very important!)

   ```
   tools/trust-origin-git-config
   ```



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
