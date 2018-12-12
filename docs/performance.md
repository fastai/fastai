---
title: Performance Tips and Tricks
---

This document will show you how to speed things up and get more out of your GPU/CPU.

## Mixed Precision Training

Combined FP16/FP32 training can tremendously improve training speed and use less GPU RAM. For theory behind it see this [thread](https://forums.fast.ai/t/mixed-precision-training/20720/3)

To deploy it see [these instructions](http://docs.fast.ai/callbacks.fp16.html).



## Faster Image Processing

### libjpeg-turbo - faster compression/decompression libjpeg replacement

[`libjpeg-turbo`](https://libjpeg-turbo.org/) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec). On x86 platforms it accelerates baseline JPEG compression and decompression and progressive JPEG compression. `libjpeg-turbo` is generally 2-6x as fast as libjpeg, all else being equal.

When you install it system-wide it provides a drop-in replacement for the `libjpeg` library. Some packages that rely on this library will be able to start using it right away, others may need to be recompiled against the replacement library.

Here is its [git-repo](https://github.com/libjpeg-turbo/libjpeg-turbo).

The `Pillow-SIMD` entry provides extra information about `libjpeg-turbo`.


### Pillow-SIMD

There is a faster `Pillow` version out there.

#### Background

First, there was PIL (Python Image Library). And then its development was abandoned.

Then, [Pillow](https://github.com/python-pillow/Pillow/) forked PIL as a drop-in replacement and according to its [benchmarks](https://python-pillow.org/pillow-perf/) it is significantly faster than `ImageMagick`, `OpenCV`, `IPP` and other fast image processing libraries (on identical hardware/platform).

Relatively recently, [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) was born to be a drop-in replacement for Pillow. This library in its turn is 4-6 times faster than Pillow, according to the same [benchmarks](https://python-pillow.org/pillow-perf/). `Pillow-SIMD` is highly optimized for common image manipulation instructions using Single Instruction, Multiple Data ([SIMD](https://en.wikipedia.org/wiki/SIMD) approach, where multiple data points are processed simultaneously. This is not parallel processing (think threads), but a single instruction processing, supported by CPU, via [data-level parallelism](https://en.wikipedia.org/wiki/Data_parallelism), similar to matrix operations on GPU, which also use SIMD.

`Pillow-SIMD` currently works only on the x86 platform. That's the main reason it's a fork of Pillow and not backported to `Pillow` - the latter is committed to support many other platforms/architectures where SIMD-support is lacking. The `Pillow-SIMD` release cycle is made so that its versions are identical Pillow's and the functionality is identical, except `Pillow-SIMD` speeds up some of them (e.g. resize).

#### Installation

To install `Pillow-SIMD`, first remove `pil`, `pillow` and `jpeg` packages:

```
conda uninstall -y --force pillow pil jpeg
pip   uninstall -y         pillow pil jpeg
```
Note, that the `--force` `conda` option forces removal of a package without removing packages that depend on it. Using this option will usually leave your `conda` environment in a broken and inconsistent state. And `pip` does it anyway. But we are going to fix your environment in the next step. Alternatively, you may choose not to use `--force`, but then it'll uninstall a whole bunch of other packages and you will need to re-install them later. It's your call.

Now we are ready to replace `libjpeg` with a drop-in replacement of `libjpeg-turbo` and then replace `Pillow` with `Pillow-SIMD`:

```
conda install -c conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
```

Since this is a forked drop-in replacement, however, the package managers don't know they have `Pillow`-like package installed, so if any update happens that triggers an update of `Pillow`, conda/pip will overwrite `Pillow-SIMD` reverting to the less speedy solution. So it's worthwhile checking your run-time setup that you're indeed using `Pillow-SIMD` in your code.

#### How to check whether you're running `Pillow` or `Pillow-SIMD`?

```
python -c "from PIL import Image; print(Image.PILLOW_VERSION)"
3.2.0.post3
```
According to the author, if `PILLOW_VERSION` has a postfix, it is `Pillow-SIMD`. (Assuming that `Pillow` will never make a `.postX` release).

#### Is JPEG compression SIMD-optimized?

`libjpeg-turbo` replacement for `libjpeg` is SIMD-optimized. In order to get `Pillow` or its faster fork `Pillow-SIMD` to use `libjpeg-turbo`, the latter needs to be already installed during the former's compilation time. Once `Pillow` is compiled/installed, it no longer matters which `libjpeg` version is installed in your virtual environment or system-wide, as long as the same `libjpeg` library remains at the same location as it was during the compilation time (it's dynamically linked).

However, if at a later time something triggers a conda or pip update on `Pillow` it will fetch a pre-compiled version which most likely is not built against `libjpeg-turbo` and replace your custom built `Pillow` or `Pillow-SIMD`.

Here is how you can see that the `PIL` library is dynamically linked to `libjpeg.so`:

```
cd ~/anaconda3/envs/pytorch-dev/lib/python3.6/site-packages/PIL/
ldd  _imaging.cpython-36m-x86_64-linux-gnu.so | grep libjpeg
        libjpeg.so.8 => ~/anaconda3/envs/pytorch-dev/lib/libjpeg.so.8
```

and `~/anaconda3/envs/pytorch-dev/lib/libjpeg.so.8` was installed by `conda install -c conda-forge libjpeg-turbo`. We know that from:

```
cd  ~/anaconda3/envs/pytorch-dev/conda-meta/
grep libjpeg.so libjpeg-turbo-2.0.1-h470a237_0.json
```

If I now install the normal `libjpeg` and do the same check on the `jpeg`'s package info:

```
conda install jpeg
cd  ~/anaconda3/envs/pytorch-dev/conda-meta/
grep libjpeg.so jpeg-9b-h024ee3a_2.json

```
I find that it's `lib/libjpeg.so.9.2.0` (`~/anaconda3/envs/pytorch-dev/lib/libjpeg.so.9.2.0`).

However, we now have an issue of the resolver showing both libraries:

```
cd ~/anaconda3/envs/pytorch-dev/lib/python3.6/site-packages/PIL/
ldd  _imaging.cpython-36m-x86_64-linux-gnu.so | grep libjpeg
        libjpeg.so.8 => ~/anaconda3/envs/pytorch-dev/lib/libjpeg.so.8
        libjpeg.so.9 => ~/anaconda3/envs/pytorch-dev/lib/libjpeg.so.9
```

And we no longer can tell which of the two will be loaded at run-time and have to inspect `/dev/<pid>/maps` instead.

Also, if `libjpeg-turbo` and `libjpeg` happen to have the same version number, even if you built `Pillow` or `Pillow-SIMD` against `libjpeg-turbo`, but then later installed the default `jpeg` with exactly the same version you will end up with the slower version.

#### How to tell whether `Pillow-SIMD` is using `libjpeg-turbo`?

It's complicated - here is some [WIP](https://github.com/python-pillow/Pillow/issues/3492).
