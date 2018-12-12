# How to contribute to fastai

First, thanks a lot for wanting to help! Make sure you have read the [doc on code style](
https://docs.fast.ai/dev/style.html) first. (Note that we don't follow PEP8, but instead follow a coding style designed specifically for numerical and interactive programming.) For help running and building the code, see the [developers guide](https://docs.fast.ai/dev/develop.html).

## Did you find a bug?

* Nobody is perfect, especially not us. But first, please double-check the bug doesn't come from something on your side. The [forum](http://forums.fast.ai/) is a tremendous source for help, and we'd advise to use it as a first step. Be sure to include as much code as you can so that other people can easily help you.
* Then, ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/fastai/fastai/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/fastai/fastai/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages as well as the result of the line `import fastai; fastai.show_install(1)`.

#### Did you write a patch that fixes a bug?

* Sign the [Contributor License Agreement](https://www.clahub.com/agreements/fastai/fastai).
* Open a new GitHub pull request with the patch.
* Ensure that your PR includes [tests](https://docs-dev.fast.ai/test) that fail without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Before submitting, please be sure you abide by our [coding style](https://docs.fast.ai/dev/style.html) and [the guide on abbreviations](https://docs.fast.ai/dev/abbr.html) and clean-up your code accordingly.

## Do you intend to add a new feature or change an existing one?

* You can suggest your change on the [fastai forum](http://forums.fast.ai/) to see if others are interested or want to help. [This topic](http://forums.fast.ai/t/fastai-v1-adding-features/23041/8) lists the features that will be added to fastai in the foreseeable future. Be sure to read it too!
* Before implementing a non-trivial new feature, first create a notebook version of your new feature, like those in [dev_nb](https://github.com/fastai/fastai_old/tree/master/dev_nb). It should show step-by-step what your code is doing, and why, with the result of each step. Try to simplify the code as much as possible. When you're happy with it, let us know on the forum (include a link to a gist with your notebook.)
* Once your approach has been discussed and confirmed on the forum, you are welcome to push a PR, including a complete description of the new feature and an example of how it's use. Be sure to document your code and read the [doc on code style](https://docs.fast.ai/dev/style.html) and [the one on abbreviations](https://docs.fast.ai/dev/abbr.html).
* Ensure that your PR includes [tests](https://docs-dev.fast.ai/test) that exercise not only your feature, but also any other code that might be impacted. Currently we have poor test coverage of existing features, so often you'll need to add tests of existing code. Your help here is much appreciated!

## How to submit notebook PRs?

* If your PR involves jupyter notebooks (`.ipynb`) you must instrument your git to `nbstripout` the notebooks, as explained [here](https://docs-dev.fast.ai/develop).

## Do you have questions about the source code?

* Please ask it on the [fastai forum](http://forums.fast.ai/) (after searching someone didn't ask the same one before with a quick search). We'd rather have the maximum of discussions there so that the largest number can benefit from it.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the `docs_src` notebook.

