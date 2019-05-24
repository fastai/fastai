---
title: Support
---

## Overview

fastai support is provided via [github issue tracker](https://github.com/fastai/fastai/issues) and the [forums](https://forums.fast.ai/).

Most issues, in particular problems with your code, should be discussed at the [forums](https://forums.fast.ai/). Either find an existing thread that's already discussing similar issues, or start a new thread.

If you are pretty sure you found a bug in the `fastai` software please submit a bug report using [github issue tracker](https://github.com/fastai/fastai/issues).

Feature requests are best discussed on the [forums](https://forums.fast.ai/).

It's always a good idea to search the forums to see whether someone else has already reported a similar issue. Usually, it will help you find a solution much faster.

If the problem is not strictly related to the `fastai` codebase, but to modules it depends on (e.g., `pytorch`, `torchvision`, `spacy`, `numpy`), often you can find solutions by searching for the error messages from the error stack trace on google or your favourite search engine.



## Reporting Issues

Before making a new issue report, please:

1.  Make sure you have the latest `conda` and/or `pip`, depending on the package manager you use:
    ```
    pip install pip -U
    conda install conda
    ```
    and then repeat the steps and see whether the problem you wanted to report still exists.

2.  Make sure [your platform is supported by `pytorch-1x`](https://github.com/fastai/fastai/blob/master/README.md#is-my-system-supported). You may have to build `pytorch` from source if it isn't.

3. Make sure you follow [the exact installation instructions](https://github.com/fastai/fastai/blob/master/README.md#installation). If you improvise and it works that's great, if it fails please RTFM ;)

4. Check the [Troubleshooting](/troubleshoot.html) document.

5. Search [forums](https://forums.fast.ai/) for a similar issues already reported.

If you still can't find a resolution, please post your issue in:

* If it's an installation issue in
[this thread](https://forums.fast.ai/t/fastai-v1-install-issues-thread/24111/1).
* For all other issue either find an existing relevant thread, or create a new one.

When you make a post, make sure to include in your post:

1. a brief summary of the problem
2. a full stack backtrace if you get an error or exception (not just the error).
3. how it can be reproduced
4. the output of the following script (including the \`\`\`text opening and closing \`\`\` so that it's formatted properly in your post):
   ```
   git clone https://github.com/fastai/fastai
   cd fastai
   python -c 'import fastai.utils; fastai.utils.show_install(1)'
   ```

   The reporting script won't work if `pytorch` wasn't installed, so if that's the case, then send in the following details:
   * output of `python --version`
   * your OS: linux/osx/windows / and linux distro+version if relevant
   * output of `nvidia-smi`  (or say CPU if none)

5. Only if it's an installation issue, the exact installation steps you followed. No need to list the installed packages, that's usually is too noisy, since it may contain hundreds of dependencies in it. Just your conda/pip install commands you did.

If the resulting output is super long, please paste it to https://pastebin.com/ and include a link to your paste, but only if it's hundreds and hundreds of lines of output - otherwise posting all the information in your post is a goodness, so that in the future other readers can compare your notes with theirs and pastebin's posts are likely to disappear.



## Do's and Don'ts

* Please do not send screenshots with stack traceback/error messages - we can't copy-n-paste from the images, instead paste them verbatim into your post.

* Code and traceback in the posts should be `code`-formatted. If you don't know markdown, you can select the snippet you want to make `code`-formatted and then hit the code button in the markdown GUI menu of the post. When you do that it will use fixed size monospaced font which makes it much easier to read.

* If your system is configured to use a non-English locale, and your error message includes non-English outcome, if possible, re-run the problematic code after running:

   `export LC_ALL=en_US.UTF-8`

    So that the error messages will be in English. You can run `locale` to see which locales you have installed.



## PRs

If you found a bug and know how to fix it, please, submit a PR with the fix [here](https://github.com/fastai/fastai/pulls).

If you'd like to contribute a new feature, please, discuss it on the [forums](https://forums.fast.ai/) first.

Make sure to read [CONTRIBUTING](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md).

Thank you.
