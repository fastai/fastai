# Entries for Issue Replies

GitHub allows for quick staple replies to the issues. Therefore please "install" the following staple replies to save yourself time, and also so that the answers by different developers are consistent.

Each developer, involved in addressing issues, needs to manually add that one by one to [their GitHub account](https://github.com/settings/replies). Once configured, you will find them in the drop-down menu accessible from the right upper corner of the comment box inside the issue page.

Note that these staple replies are global for your GitHub account, and aren't specific to `fastai` repository.

You may need to update these if the information, e.g. links, gets modified.

### fastai Entries

* fastai: sign CLA

Please sign this CLA agreement https://www.clahub.com/agreements/fastai/fastai as explained [here](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md) before we can proceed. Thank you.

* fastai: install issues [v1]

fastai 1.0.x installation issues should be reported/discussed [here](http://forums.fast.ai/t/fastai-v1-install-issues-thread/24111) instead. Thank you.

* fastai: install issues [v0]

fastai 0.7.x installation issues should be reported/discussed [here](http://forums.fast.ai/t/fastai-v0-install-issues-thread/24652) instead. Thank you.

* fastai: unstripped notebook

If your PR involves jupyter notebooks (`.ipynb`) you must instrument your git to `nbstripout` the notebooks, as explained [here](https://docs.fast.ai/dev/develop.html#things-to-run-after-git-clone). PRs with unstripped out notebooks cannot be accepted.

* fastai: how to add new functionality

If you are adding new functionality, please first create a thread on [fastai-dev](https://forums.fast.ai/c/fastai-users/fastai-dev) describing the functionality. Generally, it's best to discuss changes to functionality there first, so we can all agree on an approach. Please include a test in your PR that fails without your code, and passes with it, as well as a test of a case that already worked without your code (and still works with it). Currently, fastai has poor test coverage, so don't take the current tests as a role model - we're all working to fix it together! When creating your PR, please remove all the text in this template, and add details about your PR.
