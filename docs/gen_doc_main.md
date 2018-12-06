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
