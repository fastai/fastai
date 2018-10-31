# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional paramaters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on
Github. Parentheses after an item show the name or github id of the contributor
of that change.



## 1.0.19.dev0 (Work In Progress)

### New:

### Changed:

### Fixed:


## 1.0.18 (2018-10-30)

### Fixed:

- Fix jupyter dep version


## 1.0.17 (2018-10-30)

### New:

- Add tiny datasets

### Changed:

- remove wrong `Fbeta`

### Fixed:

- fix implementation of `fbeta`

## 1.0.16 (2018-10-30)

### New:

- `ImageDataBunch.single_from_classes` to allow single image predictions
- `DatasetBase` has `set_item` and `clear_item` to force it to always return `item`
- `DatasetBase` uses abstract `_get_x` and `_get_y`
- `batch_size` property in DeviceDataLoader
- `ClassificationLearner.predict` to get prediction on a single item
- Monkey-patched torch.Tensor so matplotlib works
- `Learner.create_unet`
- Data block API

### Changed:

- `validate` now takes optional `n_batch`
- `create_cnn` now returns a `ClassificationLearner`
- `return_path` flag to `Learner.save`
- `ImageDataBunch.show_batch()` now works for every type of dataset, removes `show_images` and `show_xy_images` as a result.
- Monkey-patched torch.utils.data.dataloader.DataLoader to create a passthrough to the dataset
- `max_workers` for `download_images`
- Change the arguments of `ObjectDetectDataset` to make it consistent with the rest of the API, changes the return of `get_annotations` to go with it.

### Fixed:

- remove empty classes in `ImageDataBunch.from_folder`

## 1.0.15 (2018-10-28)

### Breaking changes:

- `ConvLearner` ctor is replaced by a function called `create_cnn`

### New:

- `Learner` objects now determine from the loss function if there is something to add on top of the models to get the true predictions 

### Changed:

- Add `recurse` flag to `get_image_files`
- `show_xy_images` takes tensors instead of Image
- Add `classes` to SegmentationDataset
- `get_preds` now return the true probabilities
- `TTA` averages the probabilities and not the last activations of the model
- `ClassificationInterpretation` has been changed accordingly and the `sigmoid` argument has been deprecated
 
### Fixed:

- Make `pred_batch` faster and remove redundent `*`
- Bug in `Learner.pred_batch`
- Bug in `model_sizes` (thanks to dienhoa)
- Bug in `RNNLearner.classifier` when used on a multilabel dataset

## 1.0.14 (2018-10-25)

### New:

- `download_images`: multi-process download of a file or URLs
- `verify_images`: multi-process verification of directory of images with optional deletion

### Changed:

- `ImageDataBunch.from_folder` now takes `valid_pct`
- master bar support in `download_url`
- various fixes to support the latest of `fastprogress`
- `Learner.normalize()` (without args) stores calculated stats in `Learner.stats`
- `pred_batch` moved to `basic_train` and fixed for multiple inputs
- `lr_find()` prints the next step to type when completed
- New version of fastprogress used; doesn't require ipywidgets
- Removed `cifar_norm`,`cifar_denorm`,`imagenet_norm`,`imagenet_denorm`

### Fixed:


## 1.0.13 (2018-10-24)

### New: 

- pretrained language model is now downloaded directly in the .fastai/models/ folder. Use `pretrained_model=URLs.WT103`
- add an argument `stop_div` to `Learner.lr_find()` to prevent early stopping, useful for negative losses.
- add an argument `convert_mode` to `open_mask` and `SegmentationDataset` to choose the PIL conversion mode of the masks.

### Changed:

- `URLs.download_wt103()` has been removed


## 1.0.12 (2018-10-23)

### Fixed:

- change TextDataBunchClass method [`from_ids_files`, `from_tokens`, `from_df`,
  `from_csv`, `from_folder`] so that classes argument is passed to the call to TextDataset
- Strip space from file name when CSV has spaces
- Handle missing `loss_func` attr
- Pass on the `use_bn` parameter in `get_tabular_learner`
- Bad handling when final batch has size of 1 
- rolled back numpy dependency to >=1.12 (anaconda package has a upper pin on it) and to pip>=9.0.1, the old version are buggy but should be ok for fastai

## 1.0.11 (2018-10-20)

### Fixed:

- Added missing `pyyaml` dependency to conda too

### Changed:

- Use `spacy.blank` instead of `spacy.load` to avoid having to download english model

## 1.0.10 (2018-10-20)

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
