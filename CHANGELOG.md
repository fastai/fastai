# Release notes

<!-- do not remove -->

## 2.2.5

### New Features

- Enhancement: Let TextDataLoaders take in a custom `tok_text_col` ([#3208](https://github.com/fastai/fastai/pull/3208)), thanks to [@muellerzr](https://github.com/muellerzr)
- Changed dataloaders arguments to have consistent overrides ([#3178](https://github.com/fastai/fastai/pull/3178)), thanks to [@marii-moe](https://github.com/marii-moe)
- Better support for iterable datasets ([#3173](https://github.com/fastai/fastai/pull/3173)), thanks to [@jcaw](https://github.com/jcaw)

### Bugs Squashed

- BrokenProcessPool in `download_images()` on Windows ([#3196](https://github.com/fastai/fastai/issues/3196))
- error on predict() or using interp with resnet and MixUp ([#3180](https://github.com/fastai/fastai/issues/3180))
- Fix 'cat' attribute with pandas dataframe: `AttributeError: Can only use .cat accessor with a 'category' dtype` ([#3165](https://github.com/fastai/fastai/pull/3165)), thanks to [@dreamflasher](https://github.com/dreamflasher)
- `cont_cat_split` does not support pandas types ([#3156](https://github.com/fastai/fastai/issues/3156))
- `DataBlock.dataloaders` does not support the advertised "shuffle" argument ([#3133](https://github.com/fastai/fastai/issues/3133))


## 2.2.3

### New Features

- Calculate correct `nf` in `create_head` based on `concat_pool` ([#3115](https://github.com/fastai/fastai/pull/3115)), thanks to [@muellerzr](https://github.com/muellerzr)

### Bugs Squashed

- wandb integration failing with latest wandb library ([#3066](https://github.com/fastai/fastai/issues/3066))
- `Learner.load` and `LRFinder` not functioning properly for the optimizer states ([#2892](https://github.com/fastai/fastai/issues/2892))


## 2.2.2

### Bugs Squashed

- tensorboard and wandb can not access `smooth_loss` ([#3131](https://github.com/fastai/fastai/issues/3131))


## 2.2.0
### Breaking Changes

- Promote `NativeMixedPrecision` to default `MixedPrecision` (and similar for `Learner.to_fp16`); old `MixedPrecision` is now called `NonNativeMixedPrecision` ([#3127](https://github.com/fastai/fastai/issues/3127))
  - Use the new `GradientClip` callback instead of the `clip` parameter to use gradient clipping
- Adding a `Callback` which has the same name as an attribute no longer raises an exception ([#3109](https://github.com/fastai/fastai/issues/3109))
- RNN training now requires `RNNCallback`, but does not require `RNNRegularizer`; `out` and `raw_out` have moved to `RNNRegularizer` ([#3108](https://github.com/fastai/fastai/issues/3108))
  - Call `rnn_cbs` to get all callbacks needed for RNN training, optionally with regularization
- replace callback `run_after` with `order`; do not run `after` cbs on exception ([#3101](https://github.com/fastai/fastai/issues/3101))

### New Features

- Add `GradientClip` callback ([#3107](https://github.com/fastai/fastai/issues/3107))
- Make `Flatten` cast to `TensorBase` to simplify type compatibility ([#3106](https://github.com/fastai/fastai/issues/3106))
- make flattened metrics compatible with all tensor subclasses ([#3105](https://github.com/fastai/fastai/issues/3105))
- New class method `TensorBase.register_func` to register types for `__torch_function__` ([#3097](https://github.com/fastai/fastai/issues/3097))
- new `dynamic` flag for controlling dynamic loss scaling in `NativeMixedPrecision` ([#3096](https://github.com/fastai/fastai/issues/3096))
- remove need to call `to_native_fp32` before `predict`; set `skipped` in NativeMixedPrecision after NaN from dynamic loss scaling ([#3095](https://github.com/fastai/fastai/issues/3095))
- make native fp16 extensible with callbacks ([#3094](https://github.com/fastai/fastai/issues/3094))
- Calculate correct `nf` in `create_head` based on `concat_pool` ([#3115](https://github.com/fastai/fastai/pull/3115)) thanks to [@muellerzr](https://github.com/muellerzr)


## 2.1.10

### New Features

- Small DICOM segmentation dataset ([#3034](https://github.com/fastai/fastai/pull/3034)), thanks to [@moritzschwyzer](https://github.com/moritzschwyzer)

### Bugs Squashed

- `NoneType object has no attribute append` in fastbook chapter 6 BIWI example ([#3091](https://github.com/fastai/fastai/issues/3091))


## 2.1.9

### New Features

- Refactor MixUp and CutMix into MixHandler ([#3037](https://github.com/fastai/fastai/pull/3037)), thanks to [@muellerzr](https://github.com/muellerzr)
  - Refactors into a general MixHandler class, with MixUp and CutMix simply implementing a `before_batch` to perform the data augmentation. See `fastai.callback.mixup`

### Bugs Squashed

- Gradient Accumulation + Mixed Precision shows artificially high training loss ([#3048](https://github.com/fastai/fastai/issues/3048))


## 2.1.8

### New Features

### Bugs Squashed

- Update for fastcore `negate_func`->`not_`
- LR too high for gradient accumulation ([#3040](https://github.com/fastai/fastai/pull/3040)), thanks to [@marii-moe](https://github.com/marii-moe)
- Torchscript transforms incompatibility with nn.Sequential ([#2920](https://github.com/fastai/fastai/issues/2920))


## 2.1.7

### New Features

- Pytorch 1.7 subclassing support ([#2769](https://github.com/fastai/fastai/issues/2769))

### Bugs Squashed

- unsupported operand type(s) for +=: 'TensorCategory' and 'TensorText' when using AWD_LSTM for text classification ([#3027](https://github.com/fastai/fastai/issues/3027))
- UserWarning when using SaveModelCallback() on after_epoch ([#3025](https://github.com/fastai/fastai/issues/3025))
- Segmentation error: no implementation found for 'torch.nn.functional.cross_entropy' on types that implement torch_function ([#3022](https://github.com/fastai/fastai/issues/3022))
- `TextDataLoaders.from_df()` returns `TypeError: 'float' object is not iterable` ([#2978](https://github.com/fastai/fastai/issues/2978))
- Internal assert error in awd_qrnn ([#2967](https://github.com/fastai/fastai/issues/2967))


## 2.1.6

### New Features

- Option to preserve filenames in `download_images` ([#2983](https://github.com/fastai/fastai/pull/2983)), thanks to [@mess-lelouch](https://github.com/mess-lelouch)
- Deprecate `config` in `create_cnn` and instead pass kwargs directly ([#2966](https://github.com/fastai/fastai/pull/2966)), thanks to [@borisdayma](https://github.com/borisdayma)

### Bugs Squashed

- Progress and Recorder callbacks serialize their data, resulting in large Learner export file sizes ([#2981](https://github.com/fastai/fastai/issues/2981))
- `TextDataLoaders.from_df()` returns `TypeError: 'float' object is not iterable` ([#2978](https://github.com/fastai/fastai/issues/2978))
- "only one element tensors can be converted to Python scalars" exception in Siamese Tutorial ([#2973](https://github.com/fastai/fastai/issues/2973))
- Learn.load and LRFinder not functioning properly for the optimizer states ([#2892](https://github.com/fastai/fastai/issues/2892))


## 2.1.5

### Breaking Changes

- remove `log_args` ([#2954](https://github.com/fastai/fastai/issues/2954))

### New Features

- Improve performance of `RandomSplitter` (h/t @muellerzr) ([#2957](https://github.com/fastai/fastai/issues/2957))

### Bugs Squashed

- Exporting TabularLearner via learn.export() leads to huge file size ([#2945](https://github.com/fastai/fastai/issues/2945))
- `TensorPoint` object has no attribute `img_size` ([#2950](https://github.com/fastai/fastai/issues/2950))


## 2.1.4

### Breaking Changes

- moved `has_children` from `nn.Module` to free function ([#2931](https://github.com/fastai/fastai/issues/2931))

### New Features

- Support persistent workers ([#2768](https://github.com/fastai/fastai/issues/2768))

### Bugs Squashed

- `unet_learner` segmentation fails ([#2939](https://github.com/fastai/fastai/issues/2939))
- In "Transfer learning in text" tutorial, the "dls.show_batch()" show wrong outputs ([#2910](https://github.com/fastai/fastai/issues/2910))
- `Learn.load` and `LRFinder` not functioning properly for the optimizer states ([#2892](https://github.com/fastai/fastai/issues/2892))
- Documentation for `Show_Images` broken ([#2876](https://github.com/fastai/fastai/issues/2876))
- URL link for documentation for `torch_core` library from the `doc()` method gives incorrect url ([#2872](https://github.com/fastai/fastai/issues/2872))


## 2.1.3

### Bugs Squashed

- Work around broken PyTorch subclassing of some `new_*` methods ([#2769](https://github.com/fastai/fastai/issues/2769))


## 2.1.0

### New Features

- PyTorch 1.7 compatibility ([#2917](https://github.com/fastai/fastai/issues/2917))

PyTorch 1.7 includes support for tensor subclassing, so we have replaced much of our custom subclassing code with PyTorch's. We have seen a few bugs in PyTorch's subclassing feature, however, so please file an issue if you see any code failing now which was working before.

There is one breaking change in this version of fastai, which is that custom metadata is now stored directly in tensors as standard python attributes, instead of in the special `_meta` attribute. Only advanced customization of fastai OO tensors would have used this functionality, so if you do not know what this all means, then it means you did not use it.


## 2.0.19

This version was released *after* `2.1.0`, and adds fastcore 1.3 compatibility, whilst maintaining PyTorch 1.6 compatibility. It has no new features or bug fixes.


## 2.0.18

### Forthcoming breaking changes

The next version of fastai will be 2.1. It will require PyTorch 1.7, which has significant foundational changes. It should not require any code changes except for people doing sophisticated tensor subclassing work, but nonetheless we recommend testing carefully. Therefore, we recommend pinning your fastai version to `<2.1` if you are not able to fully test your fastai code when the new version comes out.

### Dependencies

- pin pytorch (`<1.7`) and torchvision (`<0.8`) requirements ([#2915](https://github.com/fastai/fastai/issues/2915))
- Add version pin for fastcore
- Remove version pin for sentencepiece


## 2.0.16

### New Features

- added support for tb projector word embeddings ([#2853](https://github.com/fastai/fastai/pull/2853)), thanks to [@floleuerer](https://github.com/floleuerer)
- Added ability to have variable length draw ([#2845](https://github.com/fastai/fastai/pull/2845)), thanks to [@marii-moe](https://github.com/marii-moe)
- add pip upgrade cell to all notebooks, to ensure colab has current fastai version ([#2843](https://github.com/fastai/fastai/issues/2843))

### Bugs Squashed

- fix TabularDataLoaders inference of cont_names to keep y_names separate ([#2859](https://github.com/fastai/fastai/pull/2859)), thanks to [@sutt](https://github.com/sutt)


## 2.0.15

### Breaking Changes

- loss functions were moved to `loss.py` ([#2843](https://github.com/fastai/fastai/pull/2810))


## 2.0.14

### New Features

- new callback event: `after_create` ([#2842](https://github.com/fastai/fastai/issues/2842))
  - This event runs after a `Learner` is constructed. It's useful for initial setup which isn't needed for every `fit`, but just once for each `Learner` (such as setting initial defaults).

- Modified XResNet to support Conv1d / Conv3d ([#2744](https://github.com/fastai/fastai/pull/2744)), thanks to [@floleuerer](https://github.com/floleuerer)
  - Supports different input dimensions, kernel sizes and stride (added parameters ndim, ks, stride). Tested with fastai_audio and fastai time series with promising results.

### Bugs Squashed

- `img_size` attribute for `TensorPoint` is not updated properly ([#2799](https://github.com/fastai/fastai/pull/2799)), thanks to [@IRailean](https://github.com/IRailean)

## 2.0.13

### Bugs Squashed

- Undo breaking num_workers fix ([#2804](https://github.com/fastai/fastai/pull/2804))
  - Some users found the recent addition of `num_workers` to inference
    functions was causing problems, particularly on Windows. This PR
    reverts that change, until we find a more reliable way to handle
    `num_workers` for inference.
- learn.tta() fails on a learner imported with load_learner() ([#2764](https://github.com/fastai/fastai/issues/2764))
- learn.summary() crashes out on 2nd transfer learning ([#2735](https://github.com/fastai/fastai/issues/2735))

## 2.0.12

### Bugs Squashed

- Undo breaking `num_workers` fix ([#2804](https://github.com/fastai/fastai/pull/2804))

## 2.0.11

### Bugs Squashed

- Fix `cont_cat_split` for multi-label classification ([#2759](https://github.com/fastai/fastai/issues/2759))
- fastbook error: "index 3 is out of bounds for dimension 0 with size 3" ([#2792](https://github.com/fastai/fastai/issues/2792))

## 2.0.10

### New Features

- update for fastcore 1.0.5 ([#2775](https://github.com/fastai/fastai/issues/2775))

## 2.0.6

### New Features

- "Remove pandas min version requirement" ([#2765](https://github.com/fastai/fastai/issues/2765))
- Modify XResNet to support Conv1d / Conv3d ([#2744](https://github.com/fastai/fastai/issues/2744))
  - Also support different input dimensions, kernel sizes and stride (added parameters ndim, ks, stride).
- Add support for multidimensional arrays for RNNDropout ([#2737](https://github.com/fastai/fastai/issues/2737))
- MCDropoutCallback to enable Monte Carlo Dropout in fastai. ([#2733](https://github.com/fastai/fastai/issues/2733))
  - A new callback to enable Monte Carlo Dropout in fastai in the `get_preds` method.
    Monte Carlo Dropout is simply enabling dropout during inference.
    Calling get_preds multiple times and stacking them yield of a distribution of predictions that you can use to evaluate your prediction uncertainty.
- adjustable workers in `get_preds` ([#2721](https://github.com/fastai/fastai/issues/2721))

## Version 2.0.0

- Initial release of v2

