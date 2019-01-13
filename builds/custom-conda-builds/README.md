# Custom Conda Recipes

Here are custom builds of dependency packages that we needed to make ourselves and upload to anaconda.org, either because they weren't there, or they weren't packaged the way we dependencies need to get resolved.

To install these packages users will need to add: -c fastai/label/test to their "conda install" command.



## torchvision

This one is built to depend on pytorch-nightly from the pytorch channel

    conda build torchvision -c pytorch --python=3.6

    anaconda upload ~/anaconda3/conda-bld/noarch/torchvision-0.2.1-pyhe7f20fa_0.tar.bz2 -u fastai --label test

then need to ask users to install torchvision=0.2.1=pyhe7f20fa_0



## dataclasses

    conda build dataclasses


## nvidia-ml-py3

    conda build nvidia-ml-py3

In case there is a new version out there, the recipe was auto-generated with: ```conda skeleton pypi nvidia-ml-py3```, but remember to add `noarch: python` inside `build:`, since by default it'll set it to 'linux-64'.


## libjpeg-turbo

See libjpeg-turbo/conda-build.txt



## libtiff

See libtiff/conda-build.txt


## pillow

See pillow/conda-build.txt


## pillow-simd

See pillow-simd/conda-build.txt



# to test installs cleanup first

    conda uninstall fastai pytorch-nightly pytorch torchvision dataclasses nvidia-ml-py3
    pip   uninstall fastai pytorch-nightly pytorch torchvision dataclasses nvidia-ml-py3
