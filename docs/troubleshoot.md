---
title: Troubleshooting
---

## Initial Installation


### Correctly configured NVIDIA drivers

Please skip this section if  your system doesn't have an NVIDIA GPU.

It doesn't matter which CUDA version you have installed on your system, always try first to install the latest `pytorch-nightly` with `cuda92` - it has all the required libraries built into the package. However, note, that you most likely will **need 396.xx+ driver for `pytorch` built with `cuda92`**. For older drivers you will probably need to install `pytorch` with `cuda90` or ever earlier.

The only thing you to need to ensure is that you have a correctly configured NVIDIA driver, which usually you can test by running: `nvidia-smi` in your console.

Please note that some users have **more than one version of NVIDIA driver installed** (e.g. one via `apt` and another from source). It's very likely that `pytorch` may have issues on such system. In such case, please, wipe any remnants of NVIDIA driver on your system, and install one NVIDIA driver of your choice.

Once you uninstalled the old drivers, to make sure you don't have any orphaned nvidia drivers on your system, usually it's enough to run:

`find /usr/ | grep libcuda.so`

or a more sweeping one (if your system updates the `mlocate` `db` regularly)

`locate -e libcuda.so`

If you get any listing as a result of  running of these commands, that means you still have some orphaned nvidia driver on your system.

Also note that `pytorch` will **silently fallback to CPU** if it reports `torch.cuda.is_available()` as `False`, so the only indicator of something being wrong will be that your notebooks will be running very slowly and you will hear your CPU revving up (if you are using a local system). Run:

```
python -c 'import fastai; fastai.show_install(1)'
```
to detect such issues. If you have this problem it'll say that your torch cuda is not available.


### Am I using my GPU(s)?

It's possible that your system is misconfigured and while you think you're using your GPU you could be running on your CPU only.

You can check that by checking the output of `torch.cuda.is_available()` - it should return `True` if `pytorch` sees your GPU(s). You can also see the state of your setup with:

```
python -c 'import fastai; fastai.show_install(1)'
```
which will include that check in its report.

But the simplest direct check is to observe the output of `nvidia-smi` while you run your code. If you don't see the process show up when you run the code, then you aren't using your GPU.

To do that you can poll `nvidia-smi`s output with either:

```
watch -n 1 nvidia-smi
```

or alternatively with:

```
nvidia-smi dmon
```

The former is useful for watching the processes, the latter provides an easier way to see usage stats as their change.

If you're on a local system, another way to tell you're not using your GPU would be an increased noise from your CPU fan.


## Installation Updates

Please skip this section unless you have post- successful install update issues.

Normally you'd update `fastai` by running `pip install -U fastai` or `conda update fastai`, using the same package manager you used to install it in first place (but in reality, either will work).

If you use the [developer setup](https://github.com/fastai/fastai/blob/master/README.md#developer-install), then you need to do a simple:

```
cd path/to/your/fastai/clone
git pull
```

Sometimes jupyter notebooks get messed up, and `git pull` might fail with an error like:
```
error: Your local changes to the following files would be overwritten by merge:
examples/cifar.ipynb
Please, commit your changes or stash them before you can merge.
Aborting
```
then either make a new `fastai` clone (the simplest), or resolve it: disable the nbstripout filter, clean up your checkout and re-enable the filter.

    tools/trust-origin-git-config -d
    git stash
    git pull
    tools/trust-origin-git-config

Of course `git stash pop` if you made some local changes and you want them back.

You can also overwrite the folder if you have no changes you want to keep with `git checkout examples` in this case, or do a reset - but you have to do it **after** you disabled the filters, and then remember to **re-enable** them back.  The instructions above do it in a non-destructive way.

The stripout filter allows us to collaborate on the notebooks w/o having conflicts with different execution counts, locally installed extensions, etc., keeping under git only the essentials. Ideally, even the `outputs` should be stripped, but that's a problem if one uses the notebooks for demo, as it is the case with `examples` notebooks.



## Managing Multiple Installations

It's possible to have multiple `fastai` installs - usually in different conda environments. And when you do that it's easy to get lost in which environment of `fastai` you currently use.

Other than `conda activate wanted-env` here are some tips to find your way around:

Tell me which environment modules are imported from:
```
import sys
print(sys.path)
```

Tell me which `fastai` library got loaded (we want to know the exact location)
```
import sys, fastai
print(sys.modules['fastai'])
```

At times a quick hack can be used to get your first notebook working and then sorting out the setup. Say you checked out `fastai` to `/tmp/`:

    cd /tmp/
    git clone https://github.com/fastai/fastai
    cd fastai

So now you know that your *uninstalled* `fastai` is located under `/tmp/fastai/`. Next, put the following on the very top of your notebook:

    import sys
    sys.path.append("/tmp/fastai")
    import fastai

and it should just work. Now, go and sort out the rest of the installation, so that you don't need to do it for every notebook.


## Support

Before making a new issue report, please:

1.  Make sure you have the latest `conda` and/or `pip`, depending on the package manager you use:
    ```
    pip install pip -U
    conda update conda
    ```
    and then check whether the problem you wanted to report still exists.

2.  Make sure [your platform is supported by the preview build of `pytorch-1.0.0`](https://github.com/fastai/fastai/blob/master/README.md#is-my-system-supported). You may have to build `pytorch` from source if it isn't.

3. Make sure you follow [the exact installation instructions](https://github.com/fastai/fastai/blob/master/README.md). If you improvise and it works that's great, if it fails please RTFM ;)

If you followed the steps in this document and couldn't find a resolution, please post a comment in this [thread](http://forums.fast.ai/t/fastai-v1-install-issues-thread/24111/1).


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

#### Do's and Don'ts:

* please do not send screenshots with trace/error messages - we can't copy-n-paste from the images, instead paste them verbatim into your post and use the markdown gui menu so that it's code-formatted.

* If your system is configured to use a non-English locale, if possible, re-run the problematic code after running:

   `export LC_ALL=en_US.UTF-8`

    So that the error messages will be in English. You can run `locale` to see which locales you have installed.

#### Bug Reports and PRs

If you found a bug and know how to fix it please submit a PR with the fix [here](https://github.com/fastai/fastai/pulls). Thank you.
