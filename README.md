# Welcome to fastai v2
> NB: This is still in early development. Use v1 unless you want to contribute to the next version of fastai


To learn more about the library, read our introduction in the [paper](https://arxiv.org/abs/2002.04688) presenting it.

## Installing

You can get all the necessary dependencies by simply installing fastai v1: `conda install -c fastai -c pytorch fastai`. Or alternatively you can automatically install the dependencies into a new environment:

```bash
git clone https://github.com/fastai/fastai2
cd fastai2
conda env create -f environment.yml
source activate fastai2
```

Then, you can install fastai v2 with pip: `pip install fastai2`. 

Or you can use an editable install (which is probably the best approach at the moment, since fastai v2 is under heavy development):
``` 
git clone https://github.com/fastai/fastai2
cd fastai2
pip install -e ".[dev]"
``` 
You should also use an editable install of [`fastcore`](https://github.com/fastai/fastcore) to go with it.

If you want to browse the notebooks and build the library from them you will need nbdev:
``` 
pip install nbdev
``` 

To use `fastai2.medical.imaging` you'll also need to:

```bash
conda install pyarrow
pip install pydicom kornia opencv-python scikit-image
```

## Tests

To run the tests in parallel, launch:

```bash
nbdev_test_nbs
```
or 
```bash
make test
```

## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The script `nbdev_diff_nbs` can let you know if there is a difference between the local library and the notebooks.
* If you made a change to the notebooks in one of the exported cells, you can export it to the library with `nbdev_build_lib` or `make fastai2`.
* If you made a change to the library, you can export it back to the notebooks with `nbdev_update_lib`.

