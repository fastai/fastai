---
title: Notes For Developers
---

### Things to Run After git clone

Make sure you follow this recipe:

    git clone https://github.com/fastai/fastai
    cd fastai
    tools/run-after-git-clone

This will take care of everything that is explained in the following sections. That is `tools/run-after-git-clone` will execute the scripts that are explained individually below. You still need to know what they do, but you need to execute just one script.

Note: windows users, not using bash emulation, will need to invoke the command as:

    python tools\run-after-git-clone

Note that if you work on `fastai/fastai_docs` repository as well, you need to run that script once too in the directory of that repository upon cloning it.


#### after-git-clone #1: a mandatory notebook strip out

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




### Development Editable Install

For deploying the `fastai` module's files, while being able to edit them, make sure to uninstall any previously installed `fastai`:

   ```
   pip   uninstall fastai
   conda uninstall fastai
   ```

And then do an editable install:

   ```
   pip install -e .[dev]
   ```

It's almost the same as:

   ```
   pip install -e .
   ```

but the former will also install extra dependencies needed only by developers.

Best not to use `python setup.py develop` method [doc](https://setuptools.readthedocs.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode).


### Switching Conda Environments in Jupyter

Other than the normal switching environments with restarts:

   ```
   source activate env1
   jupyter notebook
   (Ctrl-C to kill jupyter)
   source activate env2
   jupyter notebook
   ```

You can install [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels), which provides a separate jupyter kernel for each conda environment, along with the appropriate code to handle their setup. This makes switching conda environments as simple as switching jupyter kernel (e.g. from the kernel menu). And you don't need to worry which environment you started `jupyter notebook` from - just choose the right environment from the notebook.


### Full Diffs Mailing List

If you'd like to follow closely the development of fastai, and you don't like clicking around github, we have a read-only full diffs mailing list that is open to all.

You can subscribe or unsubscribe at any time on your own accord [here](https://groups.google.com/forum/#!forum/fastai-diff).

If you need to comment on any diff you read on that list, click on the link on top of the email and it'll take you to the github page, corresponding to that commit, and you can comment there. Alternatively ask questions on the developer's section of the  [forums](https://forums.fast.ai/c/fastai-dev).

Make sure you use a descent email client, surely don't read it on google groups or gmail, you need to have a client that can properly render fixed fonts and not use a variable fonts which look horrible with diffs. For example it works well in [Thunderbird](https://www.thunderbird.net/).

Chances are that your email client may start putting those into the spam folder, so make sure you tell it it's ham!

You will probably want to filter these emails into a dedicated folder. If so use the `List-ID` email header in the configuration of your email:

```
List-ID: <fastai-diff.googlegroups.com>
```
