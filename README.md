# fastai2

Temporary home for fastai v2 while it's being developed. Docs are at http://dev.fast.ai .

## Installation

You can get all the necessary dependencies by simply installing fastai v1: `conda install -c fastai -c pytorch fastai`. Or alternatively you can automatically install the dependencies into a new environment:

```bash
cd fastai_dev
conda env create -f environment.yml
```

Then, you can install fastai v2 with pip: `pip install git+https://github.com/fastai/fastai2`. Or clone this repo, cd to its directory, and `pip install -e .` for an *editable install* (which is probably the best approach at the moment, since fastai v2 is under heavy development).

To use `fastai2.medical.imaging` you'll also need to:

```bash
conda install pyarrow
pip install pydicom kornia opencv-python scikit-image
```

## Tests

To run the tests in parallel, do something like this:

```bash
tools/test_nbs
```

## Contributing

After you clone this repository, please run `tools/run-after-git-clone` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

Before submitting a PR, check that the local library and notebooks match. The command `nbdev_diff_nbs` can let you know if there is a difference between the local library and the notebooks.
* If you made a change to the notebooks in one of the exported cells, you can export it to the library with `nbdev_build_lib`.
* If you made a change to the library, you can export it back to the notebooks with `nbdev_update_lib`.

