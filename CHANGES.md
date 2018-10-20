# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional paramaters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on
Github. Parentheses after an item show the name or github id of the contributor
of that change.

<!-- template
## 1.0.7dev (Work In Progress)

### New:

### Changed:

### Fixed:
-->

## 1.0.11 (2018-10-20))

### Fixed:

- Added missing `pyyaml` dependency to conda too

### Changed:

- Use `spacy.blank` instead of `spacy.load` to avoid having to download english model

## 1.0.10 (2018-10-20))

### Fixed:

- Added missing `pyyaml` dependency

## 1.0.9 (2018-10-20)

### New:

- `EarlyStoppingCallback`, `SaveModelCallback`, `TerminateOnNaNCallback`
  (initial draft: fredguth)
- `datapath4file(filename)` returns suitable path to store or find data file
  called `filename`, using config file `~/.fastai/config.yml`, and default data
  directory `~/.fastai/data`, unless `./data` exists and contains that file
- MSELossFlat() loss function
- Simple integration tests for all applications

### Changed:

- `data` is now called `basic_data` to avoid weird conflicts when naming our
  data objects data.
- `datasets.untar_data` and `datasets.download_data` will now download to
  fastai home directory `~/.fastai/data` if the dataset does not already exist
  locally `./data`.

### Fixed:

- add `dep_var` column in `test_df` if it doesn't exists (Kevin Bird)
- backwards=True when creating a LanguageModelLoader (mboyanov)

## 1.0.8 (2018-10-20)

- Not released

## 1.0.7 (2018-10-19)

### New:

- New class `ImagePoints` for targets that are a set of point coordinates
- New function `Image.predict(learn:Learner)` to get the activations of the
  model in `Learner` for an image
- New function `Learner.validate` to validate on a given dl (default
  `valid_dl`), with maybe new metrics or callbacks
- New function `error_rate` which is just `1-accuracy()`

### Changed:

- All vision models are now in the `models` module, including torchvision
  models (where tested and supported). So use `models` instead of `tvm` now. If
  your preferred torchvision model isn't imported, feel free to test it out and
  tell us on the forum if it works. And if it doesn't, a PR with a test and a fix
  would be appreciated!
- `ImageBBox` is now a subclass of `ImagePoints`
- All metrics are now `Callback`. You can pass a regular function like
  `accuracy` that will get averaged over batch or a full `Callback` that can do
  more complex things
- All datasets convenience functions and paths are inside the `URLs` class
- `URLs` that are a sample have name now suffixed with `_SAMPLE`

### Fixed:

- Fix `WeightDropout` in RNNs when `p=0`
- `pad_collate` gets its `kwargs` from `TextClasDataBunch`
- Add small `eps` to `std` in `TabularDataset` to avoid division by zero
- `fit_one_cycle` doesn't take other callbacks
- Many broken docs links fixed

## 1.0.6 (2018-10-01)

- Last release without CHANGES updates

