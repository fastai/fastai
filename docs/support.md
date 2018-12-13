---
title: Support
---


## Support

Before making a new issue report, please:

1.  Make sure you have the latest `conda` and/or `pip`, depending on the package manager you use:
    ```
    pip install pip -U
    conda install conda
    ```
    and then repeat the steps and see whether the problem you wanted to report still exists.

2.  Make sure [your platform is supported by the preview build of `pytorch-1.0.0`](https://github.com/fastai/fastai/blob/master/README.md#is-my-system-supported). You may have to build `pytorch` from source if it isn't.

3. Make sure you follow [the exact installation instructions](https://github.com/fastai/fastai/blob/master/README.md#installation). If you improvise and it works that's great, if it fails please RTFM ;)

If you followed the steps in this document and couldn't find a resolution, please post a comment in this [thread](https://forums.fast.ai/t/fastai-v1-install-issues-thread/24111/1).


If the issue is still relevant, make sure to include in your post:

1. the output of the following script (including the \`\`\`text opening and closing \`\`\` so that it's formatted properly in your post):
   ```
   git clone https://github.com/fastai/fastai
   cd fastai
   python -c 'import fastai; fastai.show_install(1)'
   ```

   If you already have a `fastai` checkout, then just update it first:
   ```
   cd fastai
   git pull
   python -c 'import fastai; fastai.show_install(1)'
   ```

   The reporting script won't work if `pytorch` wasn't installed, so if that's the case, then send in the following details:
   * output of `python --version`
   * your OS: linux/osx/windows / and linux distro+version if relevant
   * output of `nvidia-smi`  (or say CPU if none)

2. a brief summary of the problem
3. the exact installation steps you followed

If the resulting output is very long, please paste it to https://pastebin.com/ and include a link to your paste

### Do's and Don'ts:

* please do not send screenshots with trace/error messages - we can't copy-n-paste from the images, instead paste them verbatim into your post and use the markdown gui menu so that it's code-formatted.

* If your system is configured to use a non-English locale, if possible, re-run the problematic code after running:

   `export LC_ALL=en_US.UTF-8`

    So that the error messages will be in English. You can run `locale` to see which locales you have installed.

### Bug Reports and PRs

If you found a bug and know how to fix it please submit a PR with the fix [here](https://github.com/fastai/fastai/pulls).

Make sure to read [CONTRIBUTING](https://github.com/fastai/fastai/blob/master/CONTRIBUTING.md).

Thank you.
