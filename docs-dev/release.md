# Making a Release

<!--ts-->

Table of Contents
-----------------

   * [Making a Release](#making-a-release)
      * [Table of Contents](#table-of-contents)
      * [Release Process](#release-process)
      * [Project Build](#project-build)
         * [Build Source distribution / Source Release](#build-source-distribution--source-release)
         * [Build Built Distribution](#build-built-distribution)
         * [Build Wheel](#build-wheel)
         * [Creating requirements.txt file by analyzing the code base](#creating-requirementstxt-file-by-analyzing-the-code-base)
      * [Project Publish](#project-publish)
      * [Prep](#prep)
      * [Test Suite](#test-suite)
      * [Publish](#publish)
         * [PyPI](#pypi)
            * [Various Helper Tools](#various-helper-tools)
            * [pip Dependencies](#pip-dependencies)
         * [Conda](#conda)
            * [Dealing with Missing Conda Packages](#dealing-with-missing-conda-packages)
            * [The Problem Of Supporting Different Architectures](#the-problem-of-supporting-different-architectures)
            * [Uploading and Testing](#uploading-and-testing)
               * [Test Release](#test-release)
               * [Re-uploading](#re-uploading)
            * [Various Helper Tools](#various-helper-tools-1)
         * [Documentation](#documentation)
         * [Support](#support)
      * [Tagging](#tagging)
         * [Rollback release commit and tag](#rollback-release-commit-and-tag)
         * [Run install tests in a fresh environment](#run-install-tests-in-a-fresh-environment)
      * [CI/CD](#cicd)
         * [Azure DevOps CI (CPU-only)](#azure-devops-ci-cpu-only)
            * [Usage](#usage)
            * [CI Builds](#ci-builds)
            * [PR Builds](#pr-builds)
            * [Modifying azure-pipelines.yml](#modifying-azure-pipelinesyml)
            * [Configuration](#configuration)
      * [Package Download Statistics](#package-download-statistics)
<!--te-->

## Release Process

WIP: this is partially pseudo-code, partially working code

The starting point is to know the sha1 of the last commit to go into the release

1. git pull sha1
2. edit setup.py and remove ".dev0" from version
   run `python setup.py --version` to update fastai/version.py (could be automated)
3. finalize CHANGES.md - version and date (could be automated)
4. make test
5. git commit setup.py fastai/version.py CHANGES.md
6. git tag with version v+version
7. make release

Then immediately start a new dev cycle:

1. edit setup.py" bump up version and add ".dev0"
   run `python setup.py --version` to update fastai/version.py  (could be automated)
2. edit CHANGES.md - copy the template and start a new entry for the new version (could be automated)
3. git commit setup.py fastai/version.py CHANGES.md




## Project Build


### Build Source distribution / Source Release

It provides metadata + source files.

It is needed for installing.

   ```
   python setup.py sdist
   ```



### Build Built Distribution

It provides metadata + pre-built files.

It only needs to be moved (usually by pip) to the correct locations on the target system.

   ```
   python setup.py bdist
   ```



### Build Wheel

This is a Built Distribution.

   ```
   python setup.py bdist_wheel
   ```

It's a ZIP-format archive with .whl extension

   ```
   {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
   ```

Note: To build all the requirements wheels (not needed for the release):

   ```
   pip wheel . -w dist
   ```



### Creating requirements.txt file by analyzing the code base

We will use 2 tools, each not finding all packages, but together they get it mostly right. So we run both and combine their results.

Install them with:

   ```
   pip install pipreqs pigar
   ```

or

   ```
   conda install pipreqs pigar -c conda-forge
   ```

And then to the mashup:

   ```
   cd fastai/fastai/
   pipreqs --savepath req1.txt .
   pigar -p req2.txt
   perl -pi -e 's| ||g' req2.txt
   cat req1.txt req2.txt | grep "##" | sort | uniq > req.txt
   ```

So this gives us `requirements.txt`-like file which can be used for pip. But we will get pip to sort things out from `setup.py`, by putting `.` inside `fastai/requirements.txt`.

Now make a list for `setup.py`'s `install_requires`:

   ```
   perl -nle '$q # chr(39); m/^(.*?)#/ && push @l, $1; END{ print join ", ", map {qq[$q$_$q]} @l}' req.txt
   ```

and use the output to update `setup.py`.

When merging make sure to not overwrite minimal version requirements, e.g. `pytorch>#0.5`. Also, you should manually clean these up since some will be deps only for doc authors or fastai library contributors; these don't need to be in the main requirements list.

Cleanup:

   ```
   rm req1.txt req2.txt req.txt
   ```

The same can be repeated for getting test requirements, just repeat the same process inside `tests` directory.



## Project Publish

## Prep

1. You need to register (free) with:

   - [PyPI](​https://pypi.org/account/register/)
   - [TestPyPI](https://test.pypi.org/account/register/)
   - [anaconda.org](​https://anaconda.org/​)

   After registration, to upload to fastai project, you will need to ask Jeremy to add your username to PyPI and anaconda.

2. Create file `~/.pypirc` with the following content:

   ```
   [distutils]
   index-servers#
   pypi
   testpypi

   [testpypi]
   repository: https://test.pypi.org/legacy/
   username: your testpypi username
   password: your testpypi password

   [pypi]
   username: your testpypi username
   password: your testpypi password
   ```

3. You can also setup your client to have transparent access to anaconda tools, see https://anaconda.org/YOURUSERNAME/settings/access (adjust the url to insert your username).

   You don't really need it, as the anaconda client cashes your credentials so you need to login only infrequently.

4. Install build tools:

   ```
   conda install conda-verify conda-build anaconda-client
   pip install twine>=1.12
   ```

## Test Suite

Before building the packages make sure the test suite runs successfully:

   ```
   make test
   ```

or:

   ```
   python setup.py test
   ```

When building a `fastai` conda package, it runs a basic `import fastai` test in a fresh environment. That's it.


## Publish

`fastai` package is distributed via [PyPI](https://pypi.org/) and [anaconda](https://anaconda.org/). Therefore we need to make two different builds and upload them to their respective servers upon a new release.

XXX: travis-ci.org as well.

### PyPI

1. Build the package (source and wheel)

   ```
   make dist
   ```

2. Publish:

   ```
   make release
   ```

   Note: PyPI won't allow re-uploading the same package filename, even if it's a minor fix. If you delete the file from pypi or test.pypi it still won't let you do it. So either a micro-level version needs to be bumped (A.B.C++) or some [post release string added](https://www.python.org/dev/peps/pep-0440/#post-releases) in `setup.py`.

3. Test that the uploaded package is found and gets installed:

   Test installation:

   ```
   pip install fastai==1.0.0b7
   ```

   XXX: May be add: `--force-reinstall` or manually remove preinstalled `fastai` first from your python installation: e.g. `python3.6/site-packages/fastai*`, run `python -m site` to find out the location.

   If the install is not working, check the state of the package: [https://pypi.org/project/fastai/](https://pypi.org/project/fastai/)




#### Various Helper Tools

* Complete Package Uninstall

   Sometimes with too many local installs/uninstalls into the same environment, especially if you nuke folders and files with `rm(1)`, things can get pretty messed up. So this can help diagnose what pip sees:

   ```
   pip show fastai
   [...]
   Name: fastai
   Version: 1.0.0b1
   Location: /some/path/to/git/clone/of/fastai
   ```

   Yet `pip` can't uninstall it:

   ```
   pip uninstall fastai
   Can't uninstall 'fastai'. No files were found to uninstall.
   ```

   `easy-install` (`pip install -e`) can make things very confusing as it may point to git checkouts that are no longer up-to-date. and you can't uninstall it. It's db is a plain text file here:

   ```
   path/to/lib/python3.6/site-packages/easy-install.pth
   ```

   so just removing the relevant path from this file will fix the problem. (or removing the whole file if you need to).

   Similarly, this is another place where it can hide:

   ```
   path/to/lib/python3.6/site-packages/fastai.egg-link
   ```


   Now running:

   ```
   pip show fastai
   ```

   shows nothing.


*. To upload to the test server, instead of the live PyPI server, use:


   ```
   twine upload --repository testpypi dist/*
   ```

   and to install from it:

   ```
   pip install --index-url https://test.pypi.org/simple/ fastai
   ```

   Doc: https://packaging.python.org/guides/using-testpypi/



#### pip Dependencies

Tools for finding out pip dependencies (direct and reversed).


* `pipdeptree`: `pip install pipdeptree`

   Print the whole tree of the installed base:
   ```
   pipdeptree -fl
   ```

   To find out why a particular package is installed (i.e. which package requires it):
   ```
   pipdeptree --reverse --packages  pillow
   ```

* `johnnydep`: `pip install johnnydep` (the tool is very slow!):

   Pretty-print a dependency tree for a Python distribution
   ```
   johnnydep spacy
   ```

   Resolve the dependency tree:
   ```
   johnnydep spacy --output-format pinned
   ```


### Conda

`conda-build` uses a build recipe `conda/meta.yaml`.

1. Check that it's valid:

   ```
   conda-build --check ./conda/
   ```

2. Build the fastai package (include the `pytorch` channel, for `torch/` dependencies, and fastai test channel for `torchvision/fastai`):

   ```
   conda-build ./conda/ -c pytorch -c fastai/label/main
   ```

   If `conda-build` fails with:

   ```
   conda_build.exceptions.DependencyNeedsBuildingError: Unsatisfiable dependencies for platform linux-64: {'dataclasses', 'jupyter_contrib_nbextensions'}
   ```

   it indicates that these packages are not in the specified via `-c` and user-pre-configured conda channels. Follow the instructions in the section `Dealing with Missing Conda Packages` and then come back to the current section and try to build again.



#### Dealing with Missing Conda Packages

Packages that are missing from conda, but available on pypi, need to built one at a time and uploaded to the `fastai` channel. For example, let's do it for the `fastprogress` package:

   ```
   conda skeleton pypi fastprogress
   conda-build fastprogress
   # the output from the above command will tell the path to the built package
   anaconda upload -u fastai ~/anaconda3/conda-bld/path/to/fastprogress-0.1.4-py36_0.tar.bz2
   ```

and then rerun `conda-build` and see if some packages are still missing. Repeat until all missing conda packages have been built and uploaded.

Note, that it's possible that a build of a certain package will fail as it'll depend on yet other packages that aren't on conda. So the (recursive) process will need to be repeated for those as well.

Once the extra packages have been built you can install them from the build directory locally:

   ```
   conda install --use-local fastprogress
   ```

Or upload them first and then install normally via `conda install`.

See `fastai/builds/custom-conda-builds` for recipes we created already.

#### The Problem Of Supporting Different Architectures

Every package we release on conda needs to be either `noarch` or we need to build a whole slew of packages for each platform we choose to support, `linux-64`, `win-64`, etc.

So far `fastai` is `noarch` (pure python), so we only need to make one `python3.6` and `python3.7` releases.

But as shown in the previous section we also have to deal with several dependencies which are not on conda. If they are `noarch`, it should be easy to release conda packages for dependencies every so often. If they are platform-specific we will have to remove them from conda dependencies and ask users to install those via pip. An easy way to check whether a package for a specific platform is available is to:

   ```
   conda search -i --platform win-64
   ```


#### Uploading and Testing

Upload to the main channel:

   ```
   anaconda upload /path/to/fastai-xxx.tar.bz2 -u fastai
   ```

To test, see that you can find it:

   ```
   conda search fastai
   ```

and then validate that the installation works correctly:

   ```
   conda install -c fastai fastai
   ```

##### Test Release

If this is just a test release that shouldn't be visible to all, add the `--label test` option, like so:

   ```
   anaconda upload /path/to/fastai-xxx.tar.bz2 -u fastai --label test
   ```

And then only those who use `-c fastai/label/test` in `conda install` command will see this package:

   ```
   conda install -c fastai/label/test fastai
   ```

Any label name can be used. If none was specified, the implicit label `main` is assigned to the package.

The label can be changed either on anaconda.org, or via it's client:

   ```
   anaconda label --copy test main
   ```

this will copy all of the test package(s) back to the `main` label. Use this one with care.


You can move individual packages from one label to another (anaconda v1.7+):

   ```
   anaconda move --from-label OLD --to-label NEW SPEC
   ```

Replace OLD with the old label, NEW with the new label, and SPEC with the package to move. SPEC can be either `user/package/version/file`, or `user/package/version` in which case it moves all files in that version. For example to move any released packages that match `fastai-1.0.5-*.tar.bz2` from the `test` label to `main` and thus making it visible to all:

   ```
   anaconda move --from-label test --to-label main fastai/fastai/1.0.5
   ```

##### Re-uploading

Note, that `anaconda` client won't let you re-upload a file with the same name, as previously uploaded one, i.e. `fastai-1.0.0-py_1.tar.bz2`, so to release an update with the same package version you either (1) use `anaconda upload --force` or (2) manually delete it from anaconda.org, or (3) create a release file with a new name, by bumping the value of `number` in `meta.yaml`.

   ```
   build:
     number: 1
   ```

Now you need to rebuild the package, and if you changed the `number` to `2`, the package will now become `'fastai-1.0.0-py_2.tar.bz2`.



#### Various Helper Tools

* To render the final `meta.yaml`:

   ```
   conda-render ./conda/
   ```

This is very useful when you do any `jinja2` template processing inside `meta.yaml` and you want to see what the final outcome is.

* Once the package is built, it can be validated:

   ```
   conda-verify path/to/package.tar.bz2
   ```

* To validate the `meta.yaml` recipe (similar to using `conda-build --check`):

   ```
   conda-verify ./conda/
   ```

* To find out the dependencies of the package:

   ```
   conda search --info -c fastai fastai
   ```

   Another hacky way to find out what the exact dependencies for a given conda package (added `-c fastai/label/test` to make it check our test package):

   ```
   conda create --dry-run --json -n dummy fastai -c fastai
   ```

* Other `conda search` tricks:

  `conda search` outputs results as following:

   ```
   conda search -c pytorch "pytorch-nightly"
   Loading channels: done
   # Name                  Version           Build                   Channel
   pytorch-nightly 0.5.0.dev20180914 py3.5_cpu_0                     pytorch
   pytorch-nightly 0.5.0.dev20180914 py3.5_cuda8.0.61_cudnn7.1.2_0   pytorch
   pytorch-nightly 0.5.0.dev20180914 py3.5_cuda9.0.176_cudnn7.1.2_0  pytorch
   pytorch-nightly 0.5.0.dev20180914 py3.5_cuda9.2.148_cudnn7.1.4_0  pytorch
   [...]
   ```

   To narrow the results, e.g. show only python3 cpu builds:

   ```
   conda search -c pytorch "pytorch-nightly[build=py3*_cpu_0]"
   ```

   and then feed it to `conda install` with specific `==version=build` after the package name, e.g. `pytorch-nightly==1.0.0.dev20180916=py3.6_cpu_0`


   To search for packages for a given system (by default, packages for your current
platform are shown):

   ```
   conda search -c pytorch "pytorch-nightly[subdir=osx-64]"
   ```

   Some of the possible platforms include `linux-32`, `linux-64`, `win-64`, `osx-64`.

   And these can be combined:

   ```
   conda search -c pytorch "pytorch-nightly[subdir=osx-64, build=py3.7*]"
   ```

   To search all packages released by user `fastai`:

   ```
   conda search -c fastai --override
   ```

   To search all packages released by user `fastai` for a specific platform, e.g. `linux-64`:

   ```
   conda search -c fastai --override --platform linux-64
   ```


### Documentation

* To figure out the nuances of the `meta.yaml` recipe writing see this [tutorial](https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html)

* `meta.yaml` is written using `jinja2` `python` templating language. [API docs](http://jinja.pocoo.org/docs/2.10/api/#high-level-api)



### Support

* [conda dev chat channel](https://gitter.im/conda/conda-build)






## Tagging

* List tags

   all tags:
   ```
   git tag
   ```

   tags matching pattern:
   ```
   git tag -l "v1.8.5*"
   ```

   by date:
   ```
   git log --tags --simplify-by-decoration --pretty="format:%ci %d"
   ```

   last tag:

   ```
   git describe --abbrev=0 --tags
   ```

* Creating tags

   To tag current checkout with tag "v1.0.5" with current date:

   ```
   git tag -a test-1.0.5 -m "test-1.0.5"
   git push --tags origin master
   ```

   To tag commit 9fceb02a with tag "v1.0.5" with current date:

   ```
   git checkout 9fceb02a
   git tag -a v1.0.5 -m "v1.0.5"
   git push --tags origin master
   git checkout master
   ```

   To tag commit 9fceb02a with tag "v1.0.5" with the date of that commit:

   ```
   git checkout 9fceb02a
   GIT_COMMITTER_DATE="$(git show --format=%aD | head -1)" git tag -a v1.0.5 -m "v1.0.5"
   git push --tags origin master
   git checkout master
   ```

   or the same without needing to `git checkout` and with typing the variables only once:

   ```
   tag="v0.1.3" commit="9fceb02a" bash -c 'GIT_COMMITTER_DATE="$(git show --format=%aD $commit)" git tag -a $tag -m $tag $commit'
   git push --tags origin master
   ```

* Delete remote tag:

   An unambiguous way:

   ```
   git push origin :refs/tags/v1.0.5
   ```

   An ambiguous way (may delete a branch if it's named the same as the tag)
   ```
   git push --delete origin v1.0.5
   ```

   Delete multiple tags:

   ```
   git push --delete origin tag1 tag2
   ```

* Delete local tag:

   ```
   git tag --delete v0.1.5
   git push --tags origin master
   ```
   This is important since if the remote tag is deleted, but the local is not, then on the next `git push --tags origin master` it will get restored in remote.


Useful scripts:

* [git-backtag](https://github.com/lucasrangit/git-bin/blob/master/git-backtag)




### Rollback release commit and tag

In case something is discovered wrong after release commit was made, here is how to rollback.

```
git reset --hard HEAD~1                        # rollback the commit
git tag -d `git describe --tags --abbrev=0`    # delete the tag
```

Careful with this as it'll reset any modified files, probably `git stash` first just in case.

Once, things were fixed, `git push`, etc...


### Run install tests in a fresh environment

While CI builds now do exactly this, it might be still useful to be able to do it manually, since CI builds are very slow to tweak and experiment with. So here is a quick copy-n-paste recipe to build one and clean it up.

```
conda create -y  python=3.6 --name fastai-py3.6
conda activate fastai-py3.6
conda install -y conda
conda install -y pip setuptools
conda install -y -c pytorch pytorch-nightly cuda92
conda install -y -c fastai torchvision-nightly
conda install -y -c fastai fastai
conda uninstall -y fastai
pip install -e .[dev]
conda deactivate
conda env remove -y --name fastai-py3.6
```

## CI/CD

### Azure DevOps CI (CPU-only)

#### Usage

All the good stuff is here: [Builds](https://dev.azure.com/fastdotai/fastai/_build?definitionId=1)

It uses `fastai/azure-pipelines.yml` script to do the testing. See notes inside the script for more details on how to modify it.

By default it runs the fastai installation and a few basic tests when either `master` gets a non-document-only push event, or PR is submitted. More details on this topic can be found in the following sections.

`[...]` options in the right upper corner, next to `Queue` hides a bunch of useful functions:

  * 'Pause builds' which may be important...
  * Status Badge MD code for the `README.md` project page

To see various stats/graphs based on tests outcome, go to [Runs](https://dev.azure.com/fastdotai/fastai/_TestManagement/Runs?runId=1&_a=runCharts)

Under Project Settings, important things are:

* [Notifications](https://dev.azure.com/fastdotai/fastai/_settings/notifications)


#### CI Builds

CI Builds are triggered every time a commit is pushed into the master (except when it's an obvious document only change commit, like a change to an `.md` file).

To trigger a manual build of go to [Builds](https://dev.azure.com/fastdotai/fastai/_build), choose Queue, choose the branch (`master`) and enter the commit hash (most likely of the latest commit). This is the way to get occasional CI builds against non-master branches.




#### PR Builds

PR Builds get triggered (1) when a new PR is submitted and (2) each time a new commit is added to that PR. It will also get triggered (3) if a closed PR gets re-opened.

If you want to manually trigger a PR Build re-run, you can click on the build status which will take you to the build page at Azure Devops and there under the "..." there is an option to Rebuild.

Note, that neither green or red status of the PR guarantees that it's so. Since the check is done at the point of the PR opening (or if new commits were added to it), it's not redone if master has changed since then. So the only way to know for sure is to force a manual rebuild for a given PR.

Currently we don't have the following enforcement enabled ([PR won't be merge-able at github](https://help.github.com/articles/about-required-status-checks/
) if the PR's build status is failed.)



#### Modifying `azure-pipelines.yml`

We now have CI builds running and therefore we shouldn't break those when need to tweak
`azure-pipelines.yml`, which often involves a lot of trial and error and meanwhile all CI builds and PRs will be broken. Not good.

Therefore **Do not modify `azure-pipelines.yml` directly in master**. Do that only in a branch, then use manual build from that branch: go to [Builds](https://dev.azure.com/fastdotai/fastai/_build), choose Queue, choose the branch and the commit hash (most likely of the latest commit) and run. Only once the pipeline is working merge it into master.

And remember to sync the branch with the master changes so that you're testing the equivalent of branch.

#### Configuration

- to [enable azure CI](https://github.com/marketplace/azure-pipelines)

- [pipelines cookbook](https://docs.microsoft.com/en-us/azure/devops/pipelines/languages/python?view=vsts)

- azure [installed automatically @github webhooks](https://github.com/fastai/fastai/settings/hooks) these push events:
* triggers CI build on every push! (ouch!):

   ```
   https://dev.azure.com/fastdotai/_apis/public/hooks/externalEvents (push)
   ```
* triggers CI build on every PR:
   ```
   https://dev.azure.com/fastdotai/_apis/public/hooks/externalEvents (pull_request)
   ```



## Package Download Statistics

How many times `fastai` was downloaded?

  * from PyPI https://pepy.tech/project/fastai
  * from Conda https://anaconda.org/fastai/fastai/files

The numbers are probably higher due to caches, CDNs, etc.
