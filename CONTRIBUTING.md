# How to contribute to fastai

First, thanks a lot for wanting to help! Make sure you have read the [doc on code style](
https://docs.fast.ai/dev/style.html) first. (Note that we don't follow PEP8, but instead follow a coding style designed specifically for numerical and interactive programming.) For help running and building the code, see the [developers guide](https://docs.fast.ai/dev/develop.html).

## Note for new contributors from Jeremy

It can be tempting to jump into a new project by questioning the stylistic decisions that have been made, such as naming, formatting, and so forth. This can be especially so for python programmers contributing to this project, which is unusual in following a number of conventions that are common in other programming communities, but not in Python. However, please don’t do this, for (amongst others) the following reasons:

- Contributing to [Parkinson’s law of triviality](https://www.wikiwand.com/en/Law_of_triviality) has negative consequences for a project. Let’s focus on deep learning!
- It’s exhausting to repeat the same discussion over and over again, especially when it’s been well documented already. When you have a question about the project, please check the pages in the docs website linked here.
- You’re likely to get a warmer welcome from the community if you start out by contributing something that’s been requested on the forum, since you’ll be solving someone’s current problem.
- If you start out by just telling us your point of view, rather than studying the background behind the decisions that have been made, you’re unlikely to be contributing anything new or useful.
- I’ve been writing code for nearly 40 years now, across dozens of languages, and other folks involved have quite a bit of experience too - the approaches used are based on significant experience and research. Whilst there’s always room for improvement, it’s much more likely you’ll be making a positive contribution if you spend a few weeks studying and working within the current framework before suggesting wholesale changes.

## How to get started

Here are some ways that you can learn a lot about the library, whilst also contributing to the community:

- Pick a class, function, or method and write tests for it. For instance, here are the tests for [fastai.core](https://github.com/fastai/fastai1/blob/master/tests/test_core.py). Adding tests for anything without good test coverage is a great way to really understand that part of the library deeply, and have in-depth conversations with the dev team about the reasoning behind decisions in the code.
- Document something that is currently undocumented. You can find them by looking for the “new methods” section in any doc notebook. Here’s a [search](https://github.com/fastai/fastai/search?q=%22new+methods%22&unscoped_q=%22new+methods%22) that lists them
- Add an example of use to the docs for something that doesn’t currently have an example of use. We’d like everything soon in the docs to include an actual piece of working code demonstrating it. Currently, we’ve largely only provided working examples for stuff higher up the abstraction ladder.

## Did you find a bug?

* Nobody is perfect, especially not us. But first, please double-check the bug doesn't come from something on your side. The [forum](http://forums.fast.ai/) is a tremendous source for help, and we'd advise to use it as a first step. Be sure to include as much code as you can so that other people can easily help you.
* Then, ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/fastai/fastai/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/fastai/fastai/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages as well as the result of the line `import fastai.test_utils; fastai.test_utils.show_install(1)`.

#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes tests that fail without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Before submitting, please be sure you abide by our [coding style](https://docs.fast.ai/dev/style.html) and [the guide on abbreviations](https://docs.fast.ai/dev/abbr.html) and clean-up your code accordingly.

## Do you intend to add a new feature or change an existing one?

* You can suggest your change on the [fastai forum](http://forums.fast.ai/) to see if others are interested or want to help. [This topic](http://forums.fast.ai/t/fastai-v1-adding-features/23041/8) lists the features that will be added to fastai in the foreseeable future. Be sure to read it too!
* Before implementing a non-trivial new feature, first create a notebook version of your new feature, like those in [dev_nb](https://github.com/fastai/fastai_docs/tree/master/dev_nb). It should show step-by-step what your code is doing, and why, with the result of each step. Try to simplify the code as much as possible. When you're happy with it, let us know on the forum (include a link to gist with your notebook.)
* Once your approach has been discussed and confirmed on the forum, you are welcome to push a PR, including a complete description of the new feature and an example of how it's used. Be sure to document your code and read the [doc on code style](https://docs.fast.ai/dev/style.html) and [the one on abbreviations](https://docs.fast.ai/dev/abbr.html).
* Ensure that your PR includes tests that exercise not only your feature, but also any other code that might be impacted. Currently we have poor test coverage of existing features, so often you'll need to add tests of existing code. Your help here is much appreciated!

## How to submit notebook PRs?

Please run [`nbdev_install_git_hooks`](https://nbdev.fast.ai/cli#nbdev_install_git_hooks) in your terminal after cloning the repository. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The script [`nbdev_diff_nbs`](https://nbdev.fast.ai/sync#nbdev_diff_nbs) can let you know if there is a difference between the local library and the notebooks.

If you made a change to the notebooks in one of the exported cells, you can export it to the library with [`nbdev_build_lib`](https://nbdev.fast.ai/export2html#nbdev_build_lib) or `make fastai`.
If you made a change to the library, you can export it back to the notebooks with [`nbdev_update_lib`](https://nbdev.fast.ai/sync#nbdev_update_lib).

Furthermore, you can run tests in parallel by launching [`nbdev_test_nbs`](https://nbdev.fast.ai/test#nbdev_test_nbs) or `make test`


## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.

* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.

* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.

* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.

* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.


### Code PRs

* If your PR is a bug fix, please also include a test that demonstrates the problem, or modifies an existing test that wasn't catching that problem already. Of course, it's not a requirement, so proceed anyway if you can't figure out how to write a test, but do try. Without having a test your fix could be lost down the road. By supplying a test, you're ensuring that your projects won't break in the future.

* Same applies for PRs that implement new features - without having a test case validating this new feature, it'd be very easy for that new feature to break in the future. A test case ensures that the feature will not break.


## Do you have questions about the source code?

* Please ask it on the [fastai forum](http://forums.fast.ai/) (after searching someone didn't ask the same one before with a quick search). We'd rather have the maximum of discussions there so that the largest number can benefit from it.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the `/nbs` directory.
* To switch the `docs` submodule to ssh, `cd docs && git remote set-url origin git@github.com:fastai/fastai-docs.git`
