# Release notes

<!-- do not remove -->

## 2.0.10

### New Features

- update for fastcore 1.0.5 ([#2775](https://api.github.com/repos/fastai/fastai/issues/2775))

## 2.0.6

### New Features

- "Remove pandas min version requirement" ([#2765](https://api.github.com/repos/fastai/fastai/issues/2765))

- Modify XResNet to support Conv1d / Conv3d ([#2744](https://api.github.com/repos/fastai/fastai/issues/2744))
  - Also support different input dimensions, kernel sizes and stride (added parameters ndim, ks, stride).

- Add support for multidimensional arrays for RNNDropout ([#2737](https://api.github.com/repos/fastai/fastai/issues/2737))

- MCDropoutCallback to enable Monte Carlo Dropout in fastai. ([#2733](https://api.github.com/repos/fastai/fastai/issues/2733))
  - A new callback to enable Monte Carlo Dropout in fastai in the `get_preds` method.
    Monte Carlo Dropout is simply enabling dropout during inference.
    Calling get_preds multiple times and stacking them yield of a distribution of predictions that you can use to evaluate your prediction uncertainty.

- adjustable workers in `get_preds` ([#2721](https://api.github.com/repos/fastai/fastai/issues/2721))

## Version 2.0.0

- Initial release of v2

