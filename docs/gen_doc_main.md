---
title: Doc Maintenance
---

## Process for contributing to the docs

If you want to help us and contribute to the docs, you just have to make modifications to the source notebooks, our scripts will then automatically convert them to HTML. There is just one script to run after cloning the fastai repo, to ensure that everything works properly. The rest of this page goes more in depth about all the functionalities this module offers, but if you just want to add a sentence or correct a typo, make a PR with the notebook changed and we'll take care of the rest.

### Thing to run after git clone

Make sure you follow this recipe:

    git clone https://github.com/fastai/fastai
    cd fastai
    tools/run-after-git-clone

This will take care of everything that is explained in the following two sections. We'll tell you what they do, but you need to execute just this one script.

Note: windows users, not using bash emulation, will need to invoke the command as:

    python tools\run-after-git-clone

If you're on windows, you also need to convert the Unix symlink between `docs_src\imgs` and `docs\imgs`. You will need to (1) remove `docs_src\imgs`, (2) execute `cmd.exe` as administrator, and (3) finally, in the `docs_src` folder, execute:

    mklink /d imgs ..\docs\imgs

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

#### after-git-clone #2: automatically updating doc notebooks to be trusted on git pull

We want the doc notebooks to be already trusted when you load them in `jupyter notebook`, so this script which should be run once upon `git clone`, will install a `git` `post-merge` hook into your local check out.

The installed hook will be executed by git automatically at the end of `git pull` only if it triggered an actual merge event and that the latter was successful.

To trust run:

    tools/trust-doc-nbs-install-hook

You don't need to run it if you run:

    tools/run-after-git-clone

To distrust run:

    rm .git/hooks/post-merge

### Validate any notebooks you're contributing to

If you were using a text editor to make changes, when you are done working on a notebook improvement, please, make sure to validate that notebook's format, by simply loading it in the jupyter notebook.

Alternatively, you could use a CLI JSON validation tool, e.g. [jsonlint](https://jsonlint.com/):

    jsonlint-php example.ipynb

but it's second best, since you may have a valid JSON, but invalid notebook format, as the latter has extra requirements on which fields are valid and which are not.

## Syncing added/updated API with docs

If you're just improving the documentation prose, and not the API documentation, just edit directly the desired source files from `docs_src/*.ipynb`, `docs/*.md` and `docs/dev/*.md`. The following instructions are only needed if you tweak the API and need to make those changes visible/up-to-date in the documentation files.

The bulk of the process and setup are explained in [gen_doc.gen_notebooks](/gen_doc.gen_notebooks.html), but its primarily purpose is for doing the heavy lifting of documenting new modules. Here you will find the minimal instructions needed to do a simple synchronization of newly added function/classes or their updates for `fastai` modules that already have corresponding documentation documents.

### Prerequisites

Install the prerequisites:
```
pip install -e .[dev]
```

Install the `Hide Input` jupyter notebook extension:

1. start `jupyter notebook`
2. go to `http://localhost:8888/nbextensions`
3. enable the `Hide Input` extension


### Initial Synchronization

Let's say we want to do some changes to the docs for `data_block.py`:

First, run an update to sync any API changes that happened before your work, but not synchronized with the docs - the code is usually ahead of the docs and the docs don't get updated all the time:

```
tools/build-docs --update-nb-links  docs_src/data_block.ipynb
```
Assuming the build was successful, commit the changes.

While you don't have to do this first, it's very helpful, since now when you re-run the build you will be able to see only the changes you introduced and not potentially hundreds of changes that have nothing to do with your modifications.

Now, you can start working on the docstrings of the new or updated functions and classes, and extra prose that you'd like to add to the documentation.


### Adding a new function/class

Say you added a method to `data_block.py`:

```
def def foobar(self, times:int=1)->'str':
    "This functions returns FooBar * times"
    return "FooBar" * times
```

While the docstring can be of any length, the fastai coding style requests that it should be no longer than 120 char long.

Any extra notes should be placed inside the corresponding entry in `.ipynb` and if it's really important some perhaps as comments in the `.py`, following the docstring.

Now run:

```
tools/build-docs --document-new-fns docs_src/data_block.ipynb
```

and if you now load `docs_src/data_block.ipynb` in jupyter notebook and scroll down to the very end of the notebook, you will find a new method entry was added under the header `New Methods - Please document or move to the undocumented section`.

Take that cell and move it up to where it belong in the document, most likely in the same position it's found in the source code `.py` file. Note that if you select this cell and hit `Toggle cell input display` (the `Hide Input` extension) in the menu, you will see something like:

```
show_doc(LabelLists.foobar)
```
If something is not displayed correctly, lookup the [show_doc](/gen_doc.nbdoc.html#show_doc) function, where you can adjust the arguments to make things look right. For example, you can pass `full_name='foobar'` argument to adjust a function name (usually helpful with functions that start with `_`, or you can pass `title_level=3` if you want it to show up at a header level 3.

When you're done tweaking the `show_doc(...)` input for that entry, remember to hit again the `Toggle cell input display` button to make it invisible, so that in the final docs site it's not displayed but the functionality generated by it does.

If you need to add any extra comments, example or instructions, create one or more cells and add what you need (markup and code if need be).

When satisfied, first make sure to save the notebook (since it auto-saves only every few minutes), and then convert this notebook into html with:

```
tools/build-docs docs_src/data_block.ipynb
```

It can be a good idea to run `git diff` to check your changes, but it might be tricky since the output format is not very human-friendly. But it'll show you if you messed something up - e.g. you deleted something unintentionally.

Finally, commit the modified `.ipynb` and the corresponding `.html` file:

```
git commit docs_src/data_block.ipynb docs/data_block.html
```
and then push the changes into the repo.

Several minutes after the push you will see the updated documents at https://docs.fast.ai/data_block.html.


### Updating an existing function/class

To take care of updating any changes in the API's arguments and docstrings that are already in the API docs, execute (we are using `data_block` as an example here):

```
tools/build-docs --update-nb-links docs_src/data_block.ipynb
```

and then as in the previous section, check the diff, commit and push.


### Creating a new documentation notebook from existing module

If a fastai.* module already exists but there is no associated documentation notebook (docs_src/*.ipynb), you can generate one by running the following:

```
tools/build-docs fastai.subpackage.module
```

This will create a skeleton documentation notebook - `docs_src/subpackage.module.ipynb`. It will populate with all the module methods. These will need to be documented.

### Borked rendering

If after `git pull` you load, e.g. `docs_src/data_block.ipynb` in jupyter notebook and you get a bunch of cryptic entries like:

```
<IPython.core.display.Markdown object>
<IPython.core.display.Markdown object>
<IPython.core.display.Markdown object>
```
instead of the API documentation entries, executing:
```
tools/build-docs --update-nb-links docs_src/data_block.ipynb
```
and then reloading the notebook fixes the problem.


## Building the documentation website

The https://docs.fast.ai website is comprised from documentation notebooks converted to `.html`, `.md` files, jekyll metadata, jekyll templates (including the sidebar).

* `.md` files are automatically converted by github pages (requires no extra action)
* the sidebar and other jekyll templates under `docs/_data/` are automatically deployed by github pages (requires no extra action)
* changes in jekyll metadata require a rebuild of the affected notebooks
* changes in `.ipynb` nbs require a rebuild of the affected notebooks


### Updating sidebar

1. edit `docs_src/sidebar/sidebar_data.py`
2. `python tools/make_sidebar.py`
3. check `docs/_data/sidebars/home_sidebar.yml`
4. `git commit docs_src/sidebar/sidebar_data.py docs/_data/sidebars/home_sidebar.yml`

[jekyll sidebar documentation](https://idratherbewriting.com/documentation-theme-jekyll/#configure-the-sidebar).

### Updating notebook metadata

In order to pass the right settings to the website version of the `docs`, each notebook has a custom entry which if you look at the source code, looks like:
```
 "metadata": {
  "jekyll": {
   "keywords": "fastai",
   "toc": "false",
   "title": "Welcome to fastai"
  },
  [...]
```
Do not edit this entry manually, or your changes will be overwritten in the next metadata update.

The only correct way to change any notebook's metadata is by opening `docs_src/jekyll_metadata.ipynb`, finding the notebook you want to change the metadata for, changing it, and running the notebook, then saving and committing it and the resulting changes.


### Updating notebooks

Use this section only when you have added a new function that you want to document, or modified an existing function.

Here is how to build/update the documentation notebooks to reflect changes in the library.


To update all modified notebooks under `docs_src` run:
```bash
python tools/build-docs
```

To update specific `*ipynb` nbs:
```bash
python tools/build-docs docs_src/notebook1.ipynb docs_src/notebook2.ipynb ...
```

To update specific `fastai.*` module:
```bash
python tools/build-docs fastai.subpackage1.module1 fastai.subpackage2.module2 ...
```

To force a rebuild of all notebooks and not just the modified ones, use the `-f` option.
```bash
python tools/build-docs -f
```

To scan a module and add any new module functions to documentation notebook:
```bash
python tools/build-docs --document-new-fns
```

To automatically append new fastai methods to their corresponding documentation notebook:
```bash
python tools/build-docs --update-nb-links
```
Use the `-h` for more options.

Alternatively, [`update_notebooks`](/gen_doc.gen_notebooks.html#update_notebooks) can be run from the notebook.

To update all notebooks under `docs_src` run:
```python
update_notebooks('.')
```

To update specific python file only:
```python
update_notebooks('gen_doc.gen_notebooks.ipynb', update_nb=True)
```

`update_nb=True` inserts newly added module methods into the docs that haven't already been documented.

Alternatively, you can update a specific module:
```python
update_notebooks('fastai.gen_doc.gen_notebooks', dest_path='fastai/docs_src')
```

### Updating html only

If you are not syncronizing the code base with its documentation, but made some manual changes to the documentation notebooks, then you don't need to update the notebooks, but just convert them to `.html`:

To convert `docs_src/*ipynb` to `docs/*html`:

* only the modified `*ipynb`:

```bash
python tools/build-docs -l
```

* specific `*ipynb`s:

```bash
python tools/build-docs -l docs_src/notebook1.ipynb docs_src/notebook2.ipynb ...
```

* force to rebuild all `*ipynb`s:

```bash
python tools/build-docs -fl
```


## Links and anchors

### Validate links and anchors

After you commit doc changes please validate that all the links and `#anchors` are correct.

If it's the first time you are about to run the link checker, install the [prerequisites](https://github.com/fastai/fastai/blob/master/tools/checklink/README.md) first.

After committing the new changes, first, wait a few minutes for github pages to sync, otherwise you'll be testing an outdated live site.

Then, do:

```
cd tools/checklink
./checklink-docs.sh
```

The script will be silent and only report problems as it finds them.

Remember, that it's testing the live website, so if you detect problems and make any changes, remember to first commit the changes and wait a few minutes before re-testing.

You can also test the site locally before committing your changes, please see: [README](https://github.com/fastai/fastai/blob/master/tools/checklink/README.md).

To test the course-v3.fast.ai site, do:
```
./checklink-course-v3.sh
```

## Working with Markdown

### Preview

If you work on markdown (.md) files it helps to be able to validate your changes so that the resulting layout is not broken. [grip](https://github.com/joeyespo/grip) seems to work quite well for this purpose (`pip install grip`). For example:

```
grip -b docs/dev/release.md
```
will open a browser with the rendered markdown as html - it uses github API, so this is exacly how it'll look on github once you commit it. And here is a handy alias:

```
alias grip='grip -b'
```
so you don't need to remember the flag.

### Markdown Tips

* If you use numbered items and their number goes beyond 9 you must switch to 4-whitespace chars indentation for the paragraphs belonging to each item. Under 9 or with \* you need 3-whitespace chars as a leading indentation.
* When building tables make sure to use `--|--` and not `--+--` to separate the headers - github will not render it properly otherwise.

## Testing site locally

Install prerequisites:

```
sudo apt install ruby-bundler
```
When running this one it will ask for your user's password (basically running a sudo operation):
```
bundle install jekyll
```

Start the website:
```
cd docs
bundle exec jekyll serve
```

it will tell you which localhost url to go to to see the site.
