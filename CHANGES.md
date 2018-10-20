# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional paramaters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on Github.

<!-- template
## 1.0.7dev (Work In Progress)

### New:

### Changed:

### Fixed:
-->

## 1.0.7 (2018-10-19)

### New:

- New class `ImagePoints` for targets that are a set of point coordinates
- New function `Image.predict(learn:Learner)` to get the activations of the model in `Learner` for an image
- New function `Learner.validate` to validate on a given dl (default `valid_dl`), with maybe new metrics or callbacks
- New function `error_rate` which is just `1-accuracy()`
- New `~/fastai` home directory, config file `~/.fastai/config.yml`, and default data directory `~/.fastai/data`

### Changed:

- `data` is now called `basic_data` to avoid weird conflicts when naming our data objects data.
- All vision models are now in the `models` module, including torchvision models (where tested and supported). So use `models` instead of `tvm` now. If your preferred torchvision model isn't imported, feel free to test it out and tell us on the forum if it works. And if it doesn't, a PR with a test and a fix would be appreciated!
- `ImageBBox` is now a subclass of `ImagePoints`
- All metrics are now `Callback`. You can pass a regular function like `accuracy` that will get averaged over batch or a full `Callback` that can do more complex things
- All datasets convenience functions and paths are inside the `URLs` class
- `URLs` that are a sample have name now suffixed with `_SAMPLE`
- `datasets.untar_data` and `datasets.download_data` will now download to fastai home directory `~/.fastai/data` if the dataset does not already exist locally `./data`.

### Fixed:

- Fix `WeightDropout` in RNNs when `p=0`
- `pad_collate` gets its `kwargs` from `TextClasDataBunch`
- Add small `eps` to `std` in `TabularDataset` to avoid division by zero
- `fit_one_cycle` doesn't take other callbacks
- Many broken docs links fixed

## 1.0.6 (2018-10-01)

- Last release without CHANGES updates

## 1.0.0 (2018-10-01)

- First release

