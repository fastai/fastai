# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional paramaters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

Note that the top-most release is changes in the unreleased master branch on
Github. Parentheses after an item show the name or github id of the contributor
of that change.


## 1.0.37.dev0 (Work In Progress)

### New:

- `SequentialEx`, `MergeLayer`, and `res_block` to more easily create resnet and densenet architectures
- `no_split` method in the data block API
- `sigmoid_range` function to scale sigmoid to given range, along with `SigmoidRange` layer
- `DataBunch` performs a sanity check after its initialization and will throw a warning if something is wrong with the data.
- More GAN stuff: `gan_critic`, `AdaptiveLoss`, `accuracy_thresh_expand`, and `GANDiscriminativeLR`
- Support for one-hot encoded labels in multiclassification problems

### Changed:

- Experimental cross-connection from raw input plus extra resblock at end of unet
- Add an execution-time check for a specific version of fastprogress (`git pull` fastai updates)
- `DataBunch.export` now serializes everything (transforms and normalization included)
- `DataBunch` now has `fix_dl` attr, which is same data as `train_dl` but without shuffle or train tfms
- `pred_batch` now has `reconstruct` param, which will reconstruct each prediction into an object
- `Learner.show_results` gives a better output for image classification tasks

### Fixed:

- Windows fixes, including:
  - Most transforms can now be used in Windows with `num_workers`>0
  - Avoid recusion error with data blocks API
  - Try to avoid default `np.int32` creation where possible
- `y_range` for unet output activation
- `Image.apply_tfms` doesn't accept any kwargs anymore
- `split_from_files` works with `from_df`/`from_csv`


## 1.0.36 (2018-12-08)

### New:

- `LabelLists.load_empty` (most useful for adding test sets for inference)


## 1.0.35 (2018-12-08)

### Changed:

- Update deps to release version of pytorch v1


## 1.0.34 (2018-12-06)

### Fixed:

- pypi wheel `dataclasses` dependency for py3.6 is there again


## 1.0.33 (2018-12-05)

### New:

- `Learner.interpret` is a shortcut to `ClassificationLearner.from_learner`.

### Changed:

- Language models now use flattened loss, instead of flattening y in data loader
- `ItemList.from_folder` now has an `include` parameter to only include certain folders

### Fixed:

- `Learner.load` won't throw an error when trying to load an optimizer state of
  the wrong size, and silently ignore that optimizer state loading


## 1.0.32 (2018-12-02)

### Changed:

- `TabularDatBunch.from_df` accepts a `test_df` argument

### Fixed:

- `LanguageLearner.predict` now returns better text predictions
- Unfreezing layers didn't create a new optimizer so the unfrozen layers weren't training
- Bug in `TextDataBunch` with a mistmatched test set was causing problems on the validation set


## 1.0.31 (2018-12-01)

### New:

- `ImageCleaner` with duplicates=True to use as a duplicate detector
- `DatasetFormatter.from_similars()` to feed the most similar indexes into `ImageCleaner`
- `chunks` to separate a Collection into smaller iterables
- `batchnorm_2d` wrapper for batchnorm with init

### Changed:

- `Learner.load` and `Learner.save` will also load/save the optimizer state
- `ImageItemList` now takes optional `convert_mode`
- `Image.show` now uses `defaults.cmap` if no `cmap` passed
- `bn` param in `conv_layer` replaced by `norm_type` which takes `NormType` enum
- unet kwargs are passed down to `conv_layer`
- `Learner.fit` no longer creates a new optimizer at each call
- Add batchnorm to end of unet
- Restore `ImageDataBunch.single_from_classes`
- `ItemList.set_item` is now a context manager, so you don't need to call `clear_item`
- Removed `ItemList.clear_item`
- Init `torch.set_num_threads(4)` to avoid OpenMP process creation overhead

### Fixed:

- `Tokenizer` wasn't using >1 thread

## 1.0.30 (2018-11-28)

### New:

- `Learner.summary`
- `add_datepart`
- `DeviceDataLoader.new` method to get a copy of a `DeviceDataLoader` while changing an attribute
- `DataBunch.batch_size` allows to change the batch size of all the dataloaders

## 1.0.29 (2018-11-27)

### Breaking changes:

- `ImageDataBunch.single_from_classes` has been removed
- `Learner.create_unet` is now called `unet_learner`

### New:

- Every type of items now has a `reconstruct` method that does the opposite of
  `ItemBase.data`: taking the tensor data and creating the object back
- `Learner.show_results` now works across applications
- `DataBunch.export`: saves the internal information (classes, vocab in text,
  processors in tabular etc) need for inference in a file named 'export.pkl'.
  You can then create an `empty_data` object by using `DataBunch.load_empty(path)`
  (where `path` points to where this 'export.pkl' file is). This also works
  across applications
- GAN and CycleGAN
- `parallel`: Run a function on every element of an array, using multiple processes
- `icnr` initializes a weight matrix with ICNR
- `PixelShuffle_ICNR` layer that combines PixelShuffle, a suitable conv2d, plus
  optional weightnorm and `(scale,scale)` blurring
- `Learner.clip_grad` convenience function for `GradientClipping` callback
- `plot_flat`, `plot_multi`, `show_multi`, `show_all`: simple functions for showing images on subplots
- `ItemList.to_text` to save items to a text file
- `ItemList.filter_by_rand` to randomly sample items
- `LabelList.transform_y` to use different transformation params for `y` (thanks for Fred Monroe)
- `LabelList.{to_df,to_csv}` to save items including labels
- `DataBunch` convenience properties: `test_ds` and `single_ds`
- `DataBunch.single_item` to convert an `ItemBase` in to a batch (tensor + dummy y)
- `Learner.pred_batch()` can now take an optional batch to predict, rather than grabbing its own
- introduce `EmptyLabel` and `EmptyLabelList`

### Changed:

- `lr_range` now divides non-final layer LRs by 10, instead of 3, when called with `slice(lr)`
- `Learner.load` now has a `strict` argument like Pytorch's `load_state_dict`
- 1cycle training now uses cosine reverse annealing instead of linear
- `conv2d` and `conv_linear` now initialize weights/bias by default
- `core.to_detach` now moves data to CPU
- `vision.models.unet` now uses `PixelShuffle_ICNR` for upsampling, with
  optional weightnorm and blurring
- `vision.models.unet` final layer now has twice as many activations
- `one_batch` moved to `DataBunch`, and can `detach` and `denorm` if requested
- `Hooks` and `Hook` can now be used as context managers
- Moved some non-image-specific functions from `vision.image` to `torch_core`
- Change `grid_sample` to downsample smoothly
- Reduce the number of hooked modules to just those required in `vision.models.unet`
- `hook_output(s)` can also hook the backward/grad now
- `bn_final` param in `TabularModel` and `create_cnn` to add batchnorm after final affine layer

### Fixed:

- factory methods of `TextDataBunch` accept `max_vocab` (thanks to jfilter)
- `vision.models.unet` now uses `eval` correctly when building model
- classes are sorted when created to avoid having them change when restarting the notebook
- fix loading issues with the test set in `TextDataBunch`
- fix random bug in `TextDataBunch.from_ids` (thanks to PiotrCzapla)


## 1.0.28 (2018-11-19)

### Breaking changes:

- `get_files` and `get_image_files` now return `Path`s relative to `path`, instead of relative to `.`
- `ItemList.items` are also relative to `path` where relevant, since `get_files` is called internally
- `create_func` is removed in the data API; subclass and change the `get` method instead (in vision, you can subclass the `open` method if you want to change how the images are opened)

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

- `tools/build-docs` and `tools/update-nbs` scripts combined into one script
- Big refactor of the data block API

### Fixed:

- `download_images` works with different kind of suffixes (thanks to fpingham)


## 1.0.22 (2018-11-09)

### Breaking changes:

- We no longer import submodule names automatically with `import *`
- Callbacks are now inside the `callbacks` namespace if you `from fastai import *`

### Changed:

- All the `DataBunch` factory method use the data block API, the factory method of `Datasets` are deprecated and will be removed in a future version

### Fixed:

- `learn.predict` fixed
- wrong dimension in dice (thanks to noklam)

## 1.0.21 (2018-11-08)

### New:

- `CSVLogger` callback (thanks to devorfu)
- Initial support for image regression problems
- If a dataset class has `learner_type` then `create_cnn` uses that type to create the `Learner`
- Introduce TaskType in `DatasetBase` to deal with single/multi-class or regression problems accross applications

### Changed:

- `datasets()` now can automatically figure out what class to use in many situations
- `download_images()` now saves images with their original extensions


## 1.0.20 (2018-11-07)

### New:

- `DataBunch.dl` replaces the various `holdout`, `is_test`, and `is_train` approaches with a single consistent enum
- `fastai.text` is fully compatible with the data block API

### Changed:

- `download_url` reads the get request with `iter_content` which is robust to 'content-length' errors. (thanks to Francisco Ingham and Zach Caceres)
- `download_url` has a timeout

### Fixed:

- `create_cnn` correctly calculates # features in body correctly for more architectures
- `TextDataset` has now two subclasses for the preprocessing steps and doesn't do that preprocesing automatically
- `TextDataBunch` doesn't save the result of preprocessing automatically, you have to use `TextDataBunch.save`
- `RNNLearner.classifier` is now `text_classifier_learner` and `RNN_Learner.language_model` is now `language_model_learner`
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

- add an argument `resize_method` that tells `apply_tfms` how to resize the image to the desired size (crop, pad, squish or no)
- all the image dataset have an `image_opener` attribute (default `open_image`) that can be changed. The `SegmentationDataset` has a `mask_opener` attribute
- `add_test` and `add_test_folder` in data block API

### Changed:

- jupyter et al no longer forced dependencies
- `verify_images` can now resize images on top of checking they're not broken
- LR finder plot now uses python scientific notation instead of math superset notation

### Fixed:

- `ImageDataBunch.from_df` doesn't change the dataframe

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
- `ImageDataBunch.show_batch()` now works for every type of dataset, removes `show_images` and `show_xy_images` as a result
- Monkey-patched torch.utils.data.dataloader.DataLoader to create a passthrough to the dataset
- `max_workers` for `download_images`
- Change the arguments of `ObjectDetectDataset` to make it consistent with the rest of the API, changes the return of `get_annotations` to go with it

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
- add an argument `stop_div` to `Learner.lr_find()` to prevent early stopping, useful for negative losses
- add an argument `convert_mode` to `open_mask` and `SegmentationDataset` to choose the PIL conversion mode of the masks

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
  data objects data
- `datasets.untar_data` and `datasets.download_data` will now download to
  fastai home directory `~/.fastai/data` if the dataset does not already exist
  locally `./data`

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
