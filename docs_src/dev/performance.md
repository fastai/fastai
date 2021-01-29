---
title: Performance Tips and Tricks
---

This document will show you how to speed things up and get more out of your GPU/CPU.

## Mixed Precision Training

Combined FP16/FP32 training can tremendously improve training speed and use less GPU RAM. For theory behind it see this [thread](https://forums.fast.ai/t/mixed-precision-training/20720/3)

To deploy it see [these instructions](/callbacks.fp16.html).


## Faster Image Processing

If you notice a bottleneck in JPEG decoding (decompression) it's enough to switch to a much faster [`libjpeg-turbo`](#libjpeg-turbo), using the normal version of `Pillow`.

If you need faster image resize, blur, alpha composition, alpha premultiplication, division by alpha, grayscale and other image manipulations you need to switch to [`Pillow-SIMD`](#pillow-simd).

At the moment this section is only relevant if you're on the x86 platform.

### libjpeg-turbo

This is a faster compression/decompression `libjpeg` drop-in replacement.

[`libjpeg-turbo`](https://libjpeg-turbo.org/) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec). On x86 platforms it accelerates baseline JPEG compression and decompression and progressive JPEG compression. `libjpeg-turbo` is generally 2-6x as fast as `libjpeg`, all else being equal.

When you install it system-wide it provides a drop-in replacement for the `libjpeg` library. Some packages that rely on this library will be able to start using it right away, most will need to be recompiled against the replacement library.

Here is its [git-repo](https://github.com/libjpeg-turbo/libjpeg-turbo).

`fastai` uses `Pillow` for its image processing and you have to rebuild `Pillow` to take advantage of `libjpeg-turbo`.

To learn how to rebuild `Pillow-SIMD` or `Pillow` with `libjpeg-turbo` see the [`Pillow-SIMD`](#pillow-simd) entry.


### Pillow-SIMD

There is a faster `Pillow` version out there.

#### Background

First, there was PIL (Python Image Library). And then its development was abandoned.

Then, [Pillow](https://github.com/python-pillow/Pillow/) forked PIL as a drop-in replacement and according to its [benchmarks](https://python-pillow.org/pillow-perf/) it is significantly faster than `ImageMagick`, `OpenCV`, `IPP` and other fast image processing libraries (on identical hardware/platform).

Relatively recently, [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) was born to be a drop-in replacement for Pillow. This library in its turn is 4-6 times faster than Pillow, according to the same [benchmarks](https://python-pillow.org/pillow-perf/). `Pillow-SIMD` is highly optimized for common image manipulation instructions using Single Instruction, Multiple Data ([SIMD](https://en.wikipedia.org/wiki/SIMD) approach, where multiple data points are processed simultaneously. This is not parallel processing (think threads), but a single instruction processing, supported by CPU, via [data-level parallelism](https://en.wikipedia.org/wiki/Data_parallelism), similar to matrix operations on GPU, which also use SIMD.

`Pillow-SIMD` currently works only on the x86 platform. That's the main reason it's a fork of Pillow and not backported to `Pillow` - the latter is committed to support many other platforms/architectures where SIMD-support is lacking. The `Pillow-SIMD` release cycle is made so that its versions are identical Pillow's and the functionality is identical, except `Pillow-SIMD` speeds up some of them (e.g. resize).

#### Installation

This section explains how to install `Pillow-SIMD` w/ `libjpeg-turbo` (but the very tricky `libjpeg-turbo` part of it is identically relevant to `Pillow` - just replace `pillow-simd` with `pillow` in the code below).

Here is the tl;dr version to install `Pillow-SIMD` w/ `libjpeg-turbo` and w/o `TIFF` support:

   ```
   conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
   pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
   conda install -yc conda-forge libjpeg-turbo
   CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
   conda install -y jpeg libtiff
   ```

Here are the detailed instructions, with an optional `TIFF` support:

1. First remove `pil`, `pillow`, `jpeg` and `libtiff` packages. Also remove 'libjpeg-tubo' if a previous version is installed:

   ```
   conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
   pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
   ```
   Both conda packages `jpeg` and `libjpeg-turbo` contain a `libjpeg.so` library.
   `jpeg`'s `libjpeg.so` library will be replaced later in these instructions with `libjpeg-turbo`'s one for the duration of the build.

   `libtiff` is linked against `libjpeg.so` library from the `jpeg` conda package, and since `Pillow` will try to link against it, we must remove it too for the duration of the build. If this is not done, `import PIL` will fail.

   Note, that the `--force` `conda` option forces removal of a package without removing packages that depend on it. Using this option will usually leave your `conda` environment in a broken and inconsistent state. And `pip` does it anyway. But we are going to fix your environment in the next step. Alternatively, you may choose not to use `--force`, but then it'll uninstall a whole bunch of other packages and you will need to re-install them later. It's your call.

2. Now we are ready to replace `libjpeg` with a drop-in replacement of `libjpeg-turbo` and then replace `Pillow` with `Pillow-SIMD`:

   ```
   conda install -yc conda-forge libjpeg-turbo
   CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
   ```
   Do note that since you're building from source, you may end up not having some of the features that come with the binary `Pillow` package if the corresponding libraries aren't available on your system during the build time. For more information see: [Building from source](https://pillow.readthedocs.io/en/latest/installation.html#building-from-source).

   If you add `-v` to the `pip install` command, you will be able to see all the details of the build, and one useful part of the output is its report of what was enabled and what not, in `PIL SETUP SUMMARY`:

   ```
    --- JPEG support available
    *** OPENJPEG (JPEG2000) support not available
    --- ZLIB (PNG/ZIP) support available
    *** LIBIMAGEQUANT support not available
    *** LIBTIFF support not available
    --- FREETYPE2 support available
    --- LITTLECMS2 support available
    *** WEBP support not available
    *** WEBPMUX support not available
   ```

3. Another nuance is `libtiff` which we removed, - and that means that `Pillow` was built without `LIBTIFF` support and will not be able to read TIFF files.

   You can safely skip this step if you don't care for TIFF files.

   This can be fixed by installing a `libtiff` library, linked against `libjpeg-turbo`.
   ```
   conda install -y -c zegami libtiff-libjpeg-turbo
   ```
   and then rebuilding `Pillow` as explained in the stage above. Only linux version of the `libtiff-libjpeg-turbo` package is available at the moment.

   XXX: The `libtiff-libjpeg-turbo` package could be outdated - it's currently only available on someone's personal channel. Alternatively, it'll need to be built from scratch. Pypi's `libtiff` package doesn't help - it doesn't place `libtiff.so` under conda environment's `lib` directory.

   The other option is to install system-wide `libjpeg-turbo` and `libtiff` linked against the former.

4. Assuming that `libjpeg-turbo` and `jpeg`'s `libjpeg.so.X.X.X` don't collide you can now reinstall back the `jpeg` package - other programs most likely need it. And `libtiff` too:

   ```
   conda install -y jpeg libtiff
   ```

5. Since this is a forked drop-in replacement, however, the package managers don't know they have `Pillow`-replacement package installed, so if any update happens that triggers an update of `Pillow`, `conda`/`pip` will overwrite `Pillow-SIMD` reverting to the less speedy `Pillow` solution. So it's worthwhile checking your run-time setup that you're indeed using `Pillow-SIMD` in your code.

   That means that every time you update the `fastai` conda package you will have to rebuild `Pillow-SIMD`.

#### How to check whether you're running `Pillow` or `Pillow-SIMD`?

```
> "python -c "import PIL; print(PIL.__version__)"
'8.1.0' or 7.0.0.post3
```
According to the author, if `PILLOW_VERSION` has a postfix, it is `Pillow-SIMD`. (Assuming that `Pillow` will never make a `.postX` release).

#### Is JPEG compression SIMD-optimized?

`libjpeg-turbo` replacement for `libjpeg` is SIMD-optimized. In order to get `Pillow` or its faster fork `Pillow-SIMD` to use `libjpeg-turbo`, the latter needs to be already installed during the former's compilation time. Once `Pillow` is compiled/installed, it no longer matters which `libjpeg` version is installed in your virtual environment or system-wide, as long as the same `libjpeg` library remains at the same location as it was during the compilation time (it's dynamically linked).

However, if at a later time something triggers a conda or pip update on `Pillow` it will fetch a pre-compiled version which most likely is not built against `libjpeg-turbo` and replace your custom built `Pillow` or `Pillow-SIMD`.

Here is how you can see that the `PIL` library is dynamically linked to `libjpeg.so`:

```
cd ~/anaconda3/envs/fastai/lib/python3.6/site-packages/PIL/
ldd  _imaging.cpython-36m-x86_64-linux-gnu.so | grep libjpeg
        libjpeg.so.8 => ~/anaconda3/envs/fastai/lib/libjpeg.so.8
```

and `~/anaconda3/envs/fastai/lib/libjpeg.so.8` was installed by `conda install -c conda-forge libjpeg-turbo`. We know that from:

```
cd  ~/anaconda3/envs/fastai/conda-meta/
grep libjpeg.so libjpeg-turbo-2.0.1-h470a237_0.json
```

If I now install the normal `libjpeg` and do the same check on the `jpeg`'s package info:

```
conda install jpeg
cd  ~/anaconda3/envs/fastai/conda-meta/
grep libjpeg.so jpeg-9b-h024ee3a_2.json

```
I find that it's `lib/libjpeg.so.9.2.0` (`~/anaconda3/envs/fastai/lib/libjpeg.so.9.2.0`).

Also, if `libjpeg-turbo` and `libjpeg` happen to have the same version number, even if you built `Pillow` or `Pillow-SIMD` against `libjpeg-turbo`, but then later replaced it with the default `jpeg` with exactly the same version you will end up with the slower version, since the linking happens at build time. But so far that risk appears to be small, as of this writing, `libjpeg-turbo` releases are in the 8.x versions, whereas `jpeg`'s are in 9.x's.

#### How to tell whether `Pillow` or `Pillow-SIMD` is using `libjpeg-turbo`?

You need `Pillow>=5.4.0` to accomplish the following (install from github until then:
`pip install git+https://github.com/python-pillow/Pillow`).

```
python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
True
```

And a version-proof check:

```
from PIL import features, Image
from packaging import version

try:    ver = Image.__version__     # PIL >= 7
except: ver = Image.PILLOW_VERSION  # PIL <  7

if version.parse(ver) >= version.parse("5.4.0"):
    if features.check_feature('libjpeg_turbo'):
        print("libjpeg-turbo is on")
    else:
        print("libjpeg-turbo is not on")
else:
    print(f"libjpeg-turbo' status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {ver}")
```

### Conda packages

The `fastai` conda (test) channel has an experimental `pillow` package built against a custom build of `libjpeg-turbo`. There are python 3.6 and 3.7 linux builds:

To install:
```
conda uninstall -y --force pillow libjpeg-turbo
conda install -c fastai/label/test pillow
```

There is also an experimental `pillow-simd-5.3.0.post0` conda package built against `libjpeg-turbo` and compiled with `avx2`. Try it only for python 3.6 on linux.
```
conda uninstall -y --force pillow libjpeg-turbo
conda install -c fastai/label/test pillow-simd
```

It probably won't work on your setup unless its CPU has the same capability as the one it was built on (Intel). So if it doesn't work, install `pillow-simd` from [source](https://github.com/uploadcare/pillow-simd#installation) instead.

Note that `pillow-simd` will get overwritten by `pillow` through update/install of any other package depending on `pillow`. You can fool `pillow-simd` into believing it is `pillow` and then it'll not get wiped out. You will have to [make a local build for that](https://github.com/fastai/fastai1/blob/master/builds/custom-conda-builds/pillow-simd/conda-build.txt).

If you have problems with these experimental packages please post [here](https://forums.fast.ai/t/performance-improvement-through-faster-software-components/32628/1), including the output of `python -m fastai.utils.check_perf` and `python -m fastai.utils.show_install` and the exact problem/errors you encountered.



## GPU Performance

See [GPU Memory Notes](/dev/gpu.html#gpu-memory-notes).
