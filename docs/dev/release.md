---
title: Making a Release
---

## Project Release Process Overview

Use this section if you know what you're doing for a quick release, otherwise first explore the sections below to understand what each `make` target does.

You can run `make help` or just `make` to find out what targets are available and what they do.

If for some reason you can't use `make`, remember that it is just a front-end for the normal `git` and other commands, so you can always make a release without using `make`.

The essence of the process is simple - bump the version number, build and upload the packages for conda and pypi. All the other steps handle various other things like tagging, testing the code base, testing the installability of the packages, etc.

Note, that the release process uses the master branch and also creates and uses a `release-$(version)` branch. That release branch remains after the release so that it's accessible from github as a release branch.

The exact order to be followed is essential.



## One Time Preparation

You can skip this step if you have done it once already on the system you're making the release from.

1. You need to register (free) with:

    - [PyPI](https://pypi.org/account/register/)
    - [TestPyPI](https://test.pypi.org/account/register/)
    - [anaconda.org](https://anaconda.org/)

    After registration, to upload to fastai project, you will need to ask Jeremy to add your username to PyPI and anaconda.

2. Create file `~/.pypirc` with the following content:

    ```
    [distutils]
    index-servers =
      pypi
      testpypi

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: your testpypi username
    password: your testpypi password

    [pypi]
    username: your pypi username
    password: your pypi password
    ```

3. You can also setup your client to have transparent access to anaconda tools, see https://anaconda.org/YOURUSERNAME/settings/access (adjust the url to insert your username).

    You don't really need it, as the anaconda client cashes your credentials so you need to login only infrequently.





## Quick Release Process

No matter which release process you follow, always remember to start with:

```
git pull
```
Otherwise it's very easy to have an outdated checkout and release an outdated version.

If however you'd like to make a release not from the `HEAD`, but from a specific commit,

```
git checkout <desired commit>
```

then **do not use the automated process**, since it resets to `master` branch. Use the step-by-step process instead, which is already instrumented for this special case. (But we could change the fully automated release to support this way too if need be).

Here is the "I'm feeling lucky" version, do not attempt unless you understand the build process.

```
make release
```

`make test`'s non-deterministic tests may decide to fail right during the release rites. It has now been moved to the head of the process, so if it fails not due to a bug but due to its unreliability, it won't affect the release process. Just rerun `make release` again.

Here is the quick version that includes all the steps w/o the explanations. If you're unfamiliar with this process use the next section instead.

```
make tools-update
make master-branch-switch && make git-not-dirty
make test
make bump && make changes-finalize
make release-branch-create && make commit-version
make master-branch-switch
make bump-dev && make changes-dev-cycle
make commit-dev-cycle-push
make prev-branch-switch && make commit-tag-push
make dist && make upload
make test-install
make backport-check
make master-branch-switch
```

If the `make backport-check` target says you need to backport, proceed to the [backporting section](#backporting-release-branch-to-master). This stage can't be fully automated since it requires you to decide what to backport if anything.

And announce the release and its changes in [Developer chat thread](https://forums.fast.ai/t/developer-chat/22363/289).



## Step-by-step Release Process

This is a one-step at a time process. If you find any difficulties scroll down to [Detailed Release Process](#detailed-release-process), which goes into many more details and explanations.

The starting point of the workflow is a dev version of the master branch. For this process we will use `1.0.6.dev0` starting point as an example.


1. check that `CHANGES.md` looks good, remove any empty sections, but don't modify the line:

   ```
   ## 1.0.12.dev0 (Work In Progress)
   ```

   The build process relies on this exact format, it will change the version number and replace `Work In Progress` with release data automatically. If you change it manually the automated process will fail. So do not.

2. install the latest tools that will be used during the build

    ```
    make tools-update            # update pip/conda build tools
    ```

3. make sure we start with master branch

    ```
    make master-branch-switch    # git checkout master
    ```

4. check-dirty - git cleanup/stash/commit so there is nothing in the way

    ```
    make git-not-dirty || echo "Commit changes before proceeding"
    ```

5. pick a starting point

    Normally, `git pull` to HEAD is fine, but it's the best to know which 'stable' <commit sha1> to use as a starting point.

    ```
    git pull
    ```
    or:
    ```
    git checkout <commit>
    ```

6. validate quality

    ```
    make test                     # py.test tests
    ```

7. start release-$(version) branch


    ```
    make bump                     # 1.0.6.dev0 => 1.0.6
    ```

The following will fix the version and the date in `CHANGES.md`, you may want to check that it looks tidy.

    ```
    make changes-finalize         # 1.0.6.dev0 (WIP) => 1.0.6 (date)
    ```

We are ready to make the new release branch:

    ```
    make release-branch-create    # git checkout -b release-1.0.6
    make commit-version           # git commit fastai/version.py
    ```

1. go back to master and bump it to the next version + .dev0


    ```
    make master-branch-switch     # git checkout master
    make bump-dev                 # 1.0.6 => 1.0.7.dev0
    ```

    Insert a new template into `CHANGES.md for the dev cycle with new version number:
    ```
    make changes-dev-cycle        # inserts new template + version
    ```

    ```
    make commit-dev-cycle-push    # git commit fastai/version.py CHANGES.md; git push
    ```

2. now we are no longer concerned with master, all the rest of the work is done on release-$(version) branch (we are using `git checkout -` here (like in `cd -`, since we no longer have the previous version)

    ```
    make prev-branch-switch       # git checkout - (i.e. release-1.0.6 branch)
    ```

3. finalize CHANGES.md (remove empty items) - version and date (could be automated)


4. git tag with version, commit and push CHANGES.md and version.py

    ```
    make commit-tag-push          # git commit CHANGES.md; git tag; git push
    ```

5. build the packages. Note that this step can take a very long time (15 mins or more). It's important that before you run it you remove or move away any large files or directories that aren't part of the release (e.g. `data`, `tmp`, `models`, and `checkpoints`), and move them back when done.

    ```
    make dist                     # make dist-pypi; make dist-conda
    ```

    This target is composed of the two individual targets listed above, so if anything goes wrong you can run them separately.

6. upload packages.

    ```
    make upload                  # make upload-pypi; make upload-conda
    ```

    This target is composed of the two individual targets listed above, so if anything goes wrong you can run them separately.

7. test uploads by installing them (telling the installers to install the exact version we uploaded). Following the upload it may take a few minutes for the servers to update their index. This target will wait for each package to become available before it will attempt to install it.

    ```
    make test-install             # pip install fastai==1.0.6; pip uninstall fastai
                                  # conda install -y -c fastai fastai==1.0.6
    ```

8. if some problems were detected during the release process, or something was committed by mistake into the release branch, and as a result changes were made to the release branch, merge those back into the master branch. Except for the version change in `fastaai/version.py`.

    1. check whether anything needs to be backported

    ```
    make backport-check
    ```

    If the `make backport-check` target says you need to backport, proceed to the [backporting section](#backporting-release-branch-to-master). This stage can't be fully automated since it requires you to decide what to backport if anything.


9. leave this branch to be indefinitely, and switch back to master, so that you won't be mistakenly committing to the release branch when you intended `master`:

    ```
    make master-branch-switch     # git checkout master
    ```

10. announce the release and its changes in [Developer chat thread](https://forums.fast.ai/t/developer-chat/22363/289).



### Backporting release Branch To master

#### Discovery Process Quick Version

Check whether there any commits besides `fastai/version.py` from the point of branching of release-1.0.6 till its HEAD. If there are then probably there are things to backport.

   ```
   make backport-check
   ```
If the result is "Nothing to backport", you're done. Otherwise proceed to the "Performing Backporting" section below.

If by any chance you switched to the master branch already, this target won't work, since it relies on `fastai/version.py` from the release branch. So you need to do it manually, by either going back to it, if it was the last one:

   ```
   git checkout -
   ```

or typing it out:

   ```
   git checkout release-1.0.6
   ```



#### Discovery Process Detailed Version

Normally you should have just one commit where `fastai/version.py` is changed, but if you applied some fixes there will be other commits. So we can't just merge the whole branch back into the master but need to cherry-pick all but the very first (version.py change commit, which `make backport-check` will already exclude from its report).


Find what needs to be backported, there are a few ways to approach it:

* find the revision at which release-$(version) branched off

    ```
    git rev-parse --short $(git merge-base master origin/release-1.0.6)
    ```

* same, but with the long commit revision

    ```
    git merge-base master origin/release-1.0.6
    ```

* get list of commits between the branching point and the HEAD of the branch

    ```
    git log  --oneline $(git merge-base master origin/release-1.0.6)..origin/release-1.0.6
    ```

* get the diff of commits between the branching point and the HEAD of the branch
    ```
    git diff $(git merge-base master origin/release-1.0.6)..origin/release-1.0.6
    ```

* alternative GUI way: checking what needs to be backported on github

    If you want to use github presentation, go to the comparison page for the tag of the release https://github.com/fastai/fastai/compare/release-1.0.6 or the same in 3 click if you don't want to manually create it:

    1. go to https://github.com/fastai/fastai
    2. select the release branch in the left upper-ish corner
    3. click 'Compare' in the right upper-ish corner

If you are trying to do this process some time after release since you remembered you didn't backport something, do the same as above but first sync your git db:

   ```
   git fetch --all # update remote info
   git branch -a   # check which branches are visible
   ```


#### Performing Backporting

Now that you looked at any changes that were applied to the release branch since it was branched, besides the version change in `fastai/version.py`, you can cherry pick the desired changes and merge them into master.

First, switch to master:

   ```
   make master-branch-switch
   ```

If `make backport-check` gave you the following output:

   ```
   !!! These commits may need to be backported:

   ab345fe conda build fix
   62091ed update release
   ```

and you decided you wanted to backport both changes, then you can do that one by one:

   ```
   git show 62091ed        # check that this is the right rev
   git cherry-pick 62091ed # merge it into the current checkout
   ```

or if there is a contiguous sequence, you can specify the start and the end (end being on top).

   ```
   git cherry-pick 62091ed..ab345fe # merge it into the current checkout
   ```

When done, complete the backporting

   ```
   git commit -m "backporting from release branch to master"
   git push
   ```





## Detailed Release Process


The following is needed if the combined release instructions are failing or better understanding is needed. So that each step can be done separately.

`fastai` package is distributed via [PyPI](https://pypi.org/) and [anaconda](https://anaconda.org/). Therefore we need to make two different builds and upload them to their respective servers upon a new release.


### Test Suite

Before building the packages make sure the test suite runs successfully:

```
make test
```

or:

```
python setup.py test
```

When building a `fastai` conda package, it runs a basic `import fastai` test in a fresh environment. That's it.




### PyPI Build and Release Details

To build a PyPI package and release it on [pypi.org/](https://pypi.org/project/fastai/):

1. Build the pip packages (source and wheel)

    ```
    make dist-pypi
    ```

2. Publish:

    ```
    make upload-pypi
    ```

    Note: PyPI won't allow re-uploading the same package filename, even if it's a minor fix. If you delete the file from pypi or test.pypi it still won't let you do it. So either a patch-level version needs to be bumped (A.B.C++) or some [post release string added](https://www.python.org/dev/peps/pep-0440/#post-releases) in `version.py`.

3. Test that the uploaded package is found and gets installed:

    Test the webpage so that the description looks correct: [https://pypi.org/project/fastai/](https://pypi.org/project/fastai/)

    Test installation:

    ```
    pip install fastai==1.0.10
    ```

    XXX: May be add: `--force-reinstall` or manually remove preinstalled `fastai` first from your python installation: e.g. `python3.6/site-packages/fastai*`, run `python -m site` to find out the location.

    If the install is not working, check the state of the package: [https://pypi.org/project/fastai/](https://pypi.org/project/fastai/)


#### Even More Details


* Build Source distribution / Source Release

    It provides metadata + source files.

    It is needed for installing.

    ```
    python setup.py sdist
    ```

    `MANIFEST.in` is in charge of what source files are included in the package. If you want to include the whole directory `tests`, but not `tests/data` for example, adjust `MANIFEST.in` to have:

    ```
    recursive-include tests *
    prune tests/data

    ```

    For more details, see [Creating a Source Distribution](https://docs.python.org/3/distutils/sourcedist.html)

*  Build Built Distribution

    It provides metadata + pre-built files.

    It only needs to be moved (usually by pip) to the correct locations on the target system.

    ```
    python setup.py bdist
    ```

* Build Wheel

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

* `setup.py` configuration:

   * [PEP 459 -- Standard Metadata Extensions for Python Software Packages](https://www.python.org/dev/peps/pep-0459/)

   * [PEP 426 -- Metadata for Python Software Packages 2.0](https://www.python.org/dev/peps/pep-0426/)

   * [Additional meta-data](https://docs.python.org/3/distutils/setupscript.html#additional-meta-data)




#### Pip Helper Tools

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


* To upload to the test server, instead of the live PyPI server, use:


    ```
    twine upload --repository testpypi dist/*
    ```

    and to install from it:

    ```
    pip install --index-url https://test.pypi.org/simple/ fastai
    ```

    Doc: https://packaging.python.org/guides/using-testpypi/






### Conda Build Details

To build a Conda package and release it on [anaconda.org](https://anaconda.org/fastai/fastai):

1. Build the fastai conda package:

    ```
    make dist-conda

    ```

2. Upload

    ```
    make upload-conda

    ```

3. Test that the uploaded package is found and gets installed:

    Test the webpage so that the description looks correct: [https://pypi.org/project/fastai/](https://pypi.org/project/fastai/)

    Test installation:

    ```
    conda install -c fastai fastai
    ```

#### More Detailed Version

`conda-build` uses a build recipe `conda/meta.yaml`.

Note, that `conda-build` recipe now relies on `sdist` generated tarball, so you need to run: `python setup.py sdist` or `make dist-pypi-sdist` if you plan on using the raw `conda-build` commands. Otherwise, `make dist-conda` already does it all for you. Basically it expects the clean tarball with source under `./dist/`.


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

    Note, that `conda-build` recipe now relies on tarball produced by `dist-pypi-sdist` target (it happens internally if you rely on `Makefile`, but if you do it without using `make`, then make sure you built the `sdist` tarball first, which is done by:

    ```
    python setup.py sdist
    ```
    which generates `dist/fastai-$version.tar.gz`, and this is what `conda-build` recipe needs. It's important to remember that if you change any files, you must rebuild the tarball, otherwise `conda-build` will be using the outdated files. If you do `make dist-conda` then it'll be taken care of automatically.



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

##### Testing Release

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



#### Conda Helper Tools

* `conda-build` useful options

   Sometimes it helps to see what `conda-build` copied into its work folder, so there is a currently not working ` --keep-old-work` option that is supposed to do that. Until it's fixed there `--dirty` which is somewhat similar, but you have clear out `/path/to/anaconda3/envs/your-env/conda-bld/` manually before using it 2nd time - if you don't it will not sync the changes in the source tree.

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

#### Documentation

* To figure out the nuances of the `meta.yaml` recipe writing see this [tutorial](https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html)

* `meta.yaml` is written using `jinja2` `python` templating language. [API docs](http://jinja.pocoo.org/docs/2.10/api/#high-level-api)



#### Support

* [conda dev chat channel](https://gitter.im/conda/conda-build)






### Version Bumping

You can either edit `fastai/version.py` and change the version number by hand.

Or run one of these `make` targets:

Target             | Function
-------------------| --------------------------------------------
bump-major         | bump major-level unless has .devX, then don't bump, but remove .devX
bump-minor         | bump minor-level unless has .devX, then don't bump, but remove .devX
bump-patch         | bump patch-level unless has .devX, then don't bump, but remove .devX
bump               | alias to bump-patch (as it's used often)
bump-major-dev     | bump major-level and add .dev0
bump-minor-dev     | bump minor-level and add .dev0
bump-patch-dev     | bump patch-level and add .dev0
bump-dev           | alias to bump-patch-dev (as it's used often)


e.g.:

```
make bump
```

We use the semver version convention w/ python adjustment to `.devX`, instead of `-devX`:

* release: `major.minor.patch`, 0.1.10
* dev or rc: `major.minor.patch.devX`, 0.1.10.dev0

Remember that master should always have `.dev0` in its version number, e.g. `0.1.10.dev0`. Only the release branch will turn it into `0.1.10`. So when a release is made, master should immediately be switched to `0.1.11.dev0`.


### Other Makefile Targets

`make clean` removes any intermediary build artifacts.

`make` will show all possible targets with a short description of what they do.




### Tagging


Tagging targets:

* List tags

    all tags:
    ```
    git tag
    ```

    tags matching pattern:
    ```
    git tag -l "1.8.5*"
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

    To tag current checkout with tag "1.0.5" with current date:

    ```
    git tag -a test-1.0.5 -m "test-1.0.5"
    git push --tags origin master
    ```

    To tag commit 9fceb02a with tag "1.0.5" with current date:

    ```
    git checkout 9fceb02a
    git tag -a v1.0.5 -m "1.0.5"
    git push --tags origin master
    git checkout master
    ```

    To tag commit 9fceb02a with tag "1.0.5" with the date of that commit:

    ```
    git checkout 9fceb02a
    GIT_COMMITTER_DATE="$(git show --format=%aD | head -1)" git tag -a v1.0.5 -m "1.0.5"
    git push --tags origin master
    git checkout master
    ```

    or the same without needing to `git checkout` and with typing the variables only once:

    ```
    tag="0.1.3" commit="9fceb02a" bash -c 'GIT_COMMITTER_DATE="$(git show --format=%aD $commit)" git tag -a $tag -m $tag $commit'
    git push --tags origin master
    ```

    To find out the hash of the last commit in a branch, to use in back-tagging:
    ```
    git log -n 1 origin/release-1.0.25
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




### Rollback Release Commit And Tag

In case something is discovered wrong after release commit was made, here is how to rollback.

```
git reset --hard HEAD~1                        # rollback the commit
git tag -d `git describe --tags --abbrev=0`    # delete the tag
```

Careful with this as it'll reset any modified files, probably `git stash` first just in case.

Once, things were fixed, `git push`, etc...


### Run Install Tests In A Fresh Environment

While CI builds now do exactly this, it might be still useful to be able to do it manually, since CI builds are very slow to tweak and experiment with. So here is a quick copy-n-paste recipe to build one and clean it up.

```
conda create -y  python=3.6 --name fastai-py3.6
conda activate fastai-py3.6
conda install -y conda
conda install -y pip setuptools
conda install -y -c pytorch pytorch-nightly cuda92
conda install -y -c fastai torchvision-nightly
conda install -c fastai fastai
conda uninstall -y fastai
pip install -e .
conda deactivate
conda env remove -y --name fastai-py3.6
```

### Package Dependencies

We need to make sure that `setup.py` sets identical dependencies to `conda/meta.yml`. It's not always possible but it should be attempted.

To find the dependencies of a given package (including the pinned versions), using `spacy` as an example:

* Conda:
   ```
   conda search --info spacy==2.0.16
   ```

* Pypi:

   Currently it can't be done without first installing the package. And you need to install `pipdeptree` that shows the complete requirements and not just the installed versions.

   ```
   pip install pipdeptree
   pip install spacy==2.0.16
   pipdeptree --packages spacy
   ```

The following sections go into pip/conda-specific tools and methods for figuring out and resolving dependencies.

#### Conda Dependencies

Here is how you can find out currently installed packages and conda dependencies:

* To find out the currently installed version of a package:

    ```
    conda list spacy
    ```

    Same, but do not show pip-only installed packages.
    ```
    conda list --no-pip spacy
    ```


* To find out the dependencies of a package:

    ```
    conda search --info spacy==2.0.16
    ```

    Narrow down to a specific platform build:

    ```
    conda search --info spacy==2.0.16=py37h962f231_0
    ```

    Also can use a wildcard:

    ```
    conda search --info spacy==2.0.16=py37*
    ```

    It supports -c channel, for packages not in a main channel

    ```
    conda search --info -c fastai fastai=1.0.6
    ```

    If version is not specified it'll show that information on all the versions it has:

    ```
    conda search --info -c fastai fastai
    ```

    Another hacky way to find out what the exact dependencies for a given conda package are:

    ```
    conda create --dry-run --json -n dummy fastai -c fastai
    ```

    Add `-c fastai/label/test` to make it check our test package.


Here is the full Conda packages version specification table:

Constraint type         | Specification       | Result
------------------------|---------------------|-----------------------------------
Fuzzy                   |numpy=1.11           |1.11.0, 1.11.1, 1.11.2, 1.11.18 etc.
Exact                   |numpy==1.11          |1.11.0
Greater than or equal to|"numpy>=1.11"        |1.11.0 or higher
OR                      |"numpy=1.11.1|1.11.3"|1.11.1, 1.11.3
AND                     |"numpy>=1.8,<2"      |1.8, 1.9, not 2.0




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




#### PyPI Dependencies

Tools for finding out currently installed packages and pip dependencies (direct and reversed).


* `pipdeptree`: (`pip install pipdeptree`)

    For a specific package:
    ```
    pipdeptree --packages  pillow
    ```
    or with more details:
    ```
    pip show pillow
    ```

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



### Creating requirements.txt File By Analyzing The Code Base

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



## CI/CD

### Azure DevOps CI (CPU-only)

#### Usage

All the good stuff is here: [Builds](https://dev.azure.com/fastdotai/fastai/_build?definitionId=1)

It uses `fastai/azure-pipelines.yml` script to do the testing. See notes inside the script for more details on how to modify it.

By default it runs the fastai installation and a few basic tests when either `master` gets a non-document-only push event, or PR is submitted. More details on this topic can be found in the following sections.

`[...]` options in the right upper corner, next to `Queue` hides a bunch of useful functions:

  * 'Pause builds' which may be important...
  * Status Badge MD code for the `README.md` project page

To see various stats/graphs based on tests outcome, go to [Test Plans] => [Runs].

Under Project Settings, important things are:

* [Notifications]


#### CI Builds

CI Builds are triggered on every `git push` to master (except when it's an obvious document only change commit, like a change to an `.md` file).


#### PR Builds

PR Builds get triggered (1) when a new PR is submitted and (2) each time a new commit is added to that PR. It will also get triggered (3) if a closed PR gets re-opened.

If you want to manually trigger a PR Build re-run, you can click on the build status which will take you to the build page at Azure Devops and there under the "..." there is an option to Rebuild.

Note, that neither green or red status of the PR guarantees that it's so. Since the check is done at the point of the PR opening (or if new commits were added to it), it's not redone if master has changed since then. So the only way to know for sure is to force a manual rebuild for a given PR.

Currently we don't have the following enforcement enabled ([PR won't be merge-able at github](https://help.github.com/articles/about-required-status-checks/
) if the PR's build status is failed.)


#### Path Filters

By default CI runs on any push made, regardless of whether it's a code or a document changed, which is a waste, so it helps to add Include/Exclude path filters.

To do that choose a build setup, go to Edit => Triggers, "Continuous Integration", check the "Override" on the right, and enable "Path filters". Important rules - paths start with `/` and if you include an Exclude filter you must also include an Include filter!!! So for example to exclude anything under /docs from triggering a build, add 2 rules:

Type    | Path specification
--------|----------------
Include | /
Exclude | /docs

Now repeat the same for "Pull request validation".

Choose 'Save', under "Save & Queue".


#### Manual Jobs

To trigger a manual build of go to [Builds](https://dev.azure.com/fastdotai/fastai/_build), choose Queue, choose the branch (`master`) and in the Commit field either nothing or enter the desired commit hash. This is the way to get occasional CI builds against non-master branches, which is useful when testing a new pipeline.


#### Scheduled Jobs

If you want to run a build as a cron-job, rather than it getting triggered by a PR or a push, add the pipeline script as normal, and then go to that build's [Edit], and then [Triggers], disable CI and PR entries and configure a scheduled entry.


#### Modifying `azure-pipelines.yml`

We now have CI builds running and therefore we shouldn't break those when need to tweak
`azure-pipelines.yml`, which often involves a lot of trial and error and meanwhile all CI builds and PRs will be broken. Not good.

Therefore **Do not modify `azure-pipelines.yml` directly in master**. Do that only in a branch, then use manual build from that branch: go to [Builds](https://dev.azure.com/fastdotai/fastai/_build), choose Queue, choose the branch and the commit hash (most likely of the latest commit) and run. Only once the pipeline is working merge it into master.

And remember to sync the branch with the master changes so that you're testing the equivalent of branch.

#### Configuration

- to [enable azure CI](https://github.com/marketplace/azure-pipelines)

- [pipelines cookbook](https://docs.microsoft.com/en-us/azure/devops/pipelines/languages/python?view=vsts)

- azure installed automatically via github's project webhooks these push events:

   * triggers CI build on every push (except when it's only doc change)!:

    ```
    https://dev.azure.com/fastdotai/_apis/public/hooks/externalEvents (push)
    ```
   * triggers CI build on every PR:
    ```
    https://dev.azure.com/fastdotai/_apis/public/hooks/externalEvents (pull_request)
    ```


#### Multiple Pipelines In The Same Repo

Currently [New] will not let you choose an alternative pipeline. So until this is fixed, let it use the default `azure-pipelines.yml`, Save and then go and Edit it and replace with a different file from the repository (and perhaps switching to a different branch if needed), using [...].

#### Support

- General Azure DevOps issues: https://developercommunity.visualstudio.com/spaces/21/index.html
- Task-specific issues: https://github.com/Microsoft/azure-pipelines-tasks/issues/
- Agent-specific issues: https://github.com/Microsoft/azure-pipelines-agent


## Package Download Statistics

How many times `fastai` was downloaded?

  * from PyPI https://pepy.tech/project/fastai
  * from Conda https://anaconda.org/fastai/fastai/files

The numbers are probably higher due to caches, CDNs, etc.
