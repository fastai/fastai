# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional paramaters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on
Github. Parentheses after an item show the name or github id of the contributor
of that change.




## 1.0.29.dev0 (Work In Progress)

### Breaking changes:

- `ImageDataBunch.single_from_classes` has been removed 

### New:

- every type of items now has a `reconstruct` method that does the opposite of `.data`: taking the tensor data and creating the object back
- `show_results` now works across applications
- introducing `data.export()` that will save the internal information (classes, vocab in text, processors in tabular etc) need for inference in a file named 'export.pkl'. You can then create an `empty_data` object by using `DataBunch.load_empty(path)` (where `path` points to where this 'export.pkl' file is). This also works across applications.
- add basic GAN functionalities

### Changed:

- `show_batch` has been internally modified to actually grab a batch then showing it

### Fixed:

- factory methods of `TextDataBunch` accept `max_vocab` (thanks to jfilter)

## 1.0.28 (2018-11-19)

### Breaking changes:

- `get_files` and `get_image_files` now return `Path`s relative to `path`, instead of relative to `.`
- `ItemList.items` are also relative to `path` where relevant, since `get_files` is called internally
- `create_func` is removed in the data API; subclass and change the `get` method instead (in vision, you can subclass the `open` method if you want to change how the images are opened).

### New:

- `Vocab` and `TabularTransform` can now be saved
- Each application has its method to create an inference learner
- `model_summary` function for standard models (thanks to @noklam)
- Added `pca` to `torch.Tensor`
- Add methods to get embeddings from `CollabLearner`

### Fixed:

- `verify_image` - now fixes files with corrupt EXIF data

## 1.0.27 (2018-11-17)

### New:

- We can add transform to `y` in the data block API
- metric fbeta for single classification (thanks to wy-q)

### Changed:

- ItemLists can now set `self.filter_missing_y` to automatically remove items from LabelLists  training set that can't be labeled
- revert xxmaj token and `deal_caps` rule

### Fixed:


## 1.0.26 (2018-11-16)

### New:

- xxmaj token and new `deal_caps` rule

### Changed:

- `Tokenizer` has `pre_rules` and `post_rules` now (for before and after tokenization)
- `mark_fields` is now default to `False`


## 1.0.25 (2018-11-16)

### New:

- `FloatList` to do regression
- Use of real neural nets in `collab`

### Changed:

- Remove `TextFilesList` as you can now use `TextList` instead
- Consistent use of `cols` / `col` in the data block API depending on if you can pass multiple columns or not
- Collab is refactored with the data block API behind the scene
- `get_collab_learner` and `get_tabular_learner` become `collab_learner` and
  `tabular_learner` for name harmonization accross applications
- `get_embedding` becomes `embedding`
- `ImageDeleter` and `ImageRelabeler` are merged into `ImageCleaner`

### Fixed:

- `show_batch` works with `rows=1`
- Pretrained language models are saved in the correct folder (.fastai/models/)
- Splitting too slow in the data block API
- Mixup losses work with predict and TTA (thanks to bharadwaj6)
- Wrong size for the added test set in the data block API (thanks to wdhorton)
- Fix to the QRNN (thanks to PiotrCzapla)

## 1.0.24 (2018-11-13)

- No changes

## 1.0.23 (2018-11-13)

### New:

- `Learner.predict` works accross applications
- `Learner.show_batch` works accross applications

### Changed:

- `tools/build-docs` and `tools/update-nbs` scripts combined into one script.
- Big refactor of the data block API

### Fixed:

- `download_images` works with different kind of suffixes (thanks to fpingham)


## 1.0.22 (2018-11-09)

### Breaking changes:

- We no longer import submodule names automatically with `import *`
- Callbacks are now inside the `callbacks` namespace if you `from fastai import *`

### Changed:

- All the `DataBunch` factory method use the data block API, the factory method of `Datasets` are deprecated and will be removed in a future version.

### Fixed:

- `learn.predict` fixed
- wrong dimension in dice (thanks to noklam)

## 1.0.21 (2018-11-08)

### New:

- `CSVLogger` callback (thanks to devorfu)
- Initial support for image regression problems.
- If a dataset class has `learner_type` then `create_cnn` uses that type to create the `Learner`.
- Introduce TaskType in `DatasetBase` to deal with single/multi-class or regression problems accross applications.

### Changed:

- `datasets()` now can automatically figure out what class to use in many situations
- `download_images()` now saves images with their original extensions


## 1.0.20 (2018-11-07)

### New:

- `DataBunch.dl` replaces the various `holdout`, `is_test`, and `is_train` approaches with a single consistent enum.
- `fastai.text` is fully compatible with the data block API.

### Changed:

- `download_url` reads the get request with `iter_content` which is robust to 'content-length' errors. (thanks to Francisco Ingham and Zach Caceres)
- `download_url` has a timeout

### Fixed:

- `create_cnn` correctly calculates # features in body correctly for more architectures
- `TextDataset` has now two subclasses for the preprocessing steps and doesn't do that preprocesing automatically.
- `TextDataBunch` doesn't save the result of preprocessing automatically, you have to use `TextDataBunch.save`.
- `RNNLearner.classifier` is now `text_classifier_learner` and `RNN_Learner.language_model` is now `language_model_learner`.
- `pil2tensor` is faster and works on more image types (thanks to kasparlund)
- Imports in the file picker widget (thanks to Hiromi)
- Batches of size 1 will be removed during training because of the issue with BatchNorm1d
- Confusion matrix show ints if `normalize=False` (default)
- `RNNLearner.get_preds` return the preds in the right order (thanks to StatisticDean)
- `num_features_model` now works with any model
- `resize_method` wasn't properly set when passed to `ImageDataBunch`
- `reset` the RNNs at the beginning of each epoch in `RNNTrainer`

## 1.0.19 (2018-11-03)

### New:

- add an argument `resize_method` that tells `apply_tfms` how to resize the image to the desired size (crop, pad, squish or no).
- all the image dataset have an `image_opener` attribute (default `open_image`) that can be changed. The `SegmentationDataset` has a `mask_opener` attribute.
- `add_test` and `add_test_folder` in data block API.

### Changed:

- jupyter et al no longer forced dependencies
- `verify_images` can now resize images on top of checking they're not broken.
- LR finder plot now uses python scientific notation instead of math superset notation

### Fixed:

- `ImageDataBunch.from_df` doesn't change the dataframe.

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
