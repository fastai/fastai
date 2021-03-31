# Release notes

<!-- do not remove -->

## 2.2.8
### Breaking Changes

- fix optimwrapper to work with param_groups (closes #2829) ([#3241](https://github.com/fastai/fastai/pull/3241)), thanks to [@tmabraham](https://github.com/tmabraham)
  - Currently, `OptimWrapper` does not support `param_groups`, preventing use of model freezing/unfreezing and discriminative learning rate. 

This PR incorporates @muellerzr's [fix](https://muellerzr.github.io/fastai_minima/optimizer.html#Differential-Learning-Rates-and-Groups-with-Pytorch-Optimizers) into `OptimWrapper`. In doing so, the usage of `OptimWrapper` is slightly changed. For example, it changes usage from:
```
def opt_func(params, lr, **kwargs): return OptimWrapper(torch.optim.SGD(params, lr=lr))
```
to
```
def opt_func(params, **kwargs): return OptimWrapper(torch.optim.SGD, params, **kwargs)
```

This PR will also solve issue #2829 as well.

### New Features

- Support discriminative learning with OptimWrapper ([#2829](https://github.com/fastai/fastai/issues/2829))
  - Currently, the following code gives error
```python
from fastai.vision.all import *

def SGD_opt(params, **kwargs): return OptimWrapper(torch.optim.SGD(params, **kwargs))

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate, opt_func=SGD_opt)
learn.fit_one_cycle(1)
```

The error is as follows:
```python
TypeError                                 Traceback (most recent call last)
<ipython-input-133-20a3ebb82957> in <module>
     10     label_func=is_cat, item_tfms=Resize(224))
     11 
---> 12 learn = cnn_learner(dls, resnet34, metrics=error_rate, opt_func=SGD_opt)
     13 learn.fit_one_cycle(1)

~/miniconda3/lib/python3.8/site-packages/fastcore/logargs.py in _f(*args, **kwargs)
     50         log_dict = {**func_args.arguments, **{f'{k} (not in signature)':v for k,v in xtra_kwargs.items()}}
     51         log = {f'{f.__qualname__}.{k}':v for k,v in log_dict.items() if k not in but}
---> 52         inst = f(*args, **kwargs) if to_return else args[0]
     53         init_args = getattr(inst, 'init_args', {})
     54         init_args.update(log)

~/miniconda3/lib/python3.8/site-packages/fastai/vision/learner.py in cnn_learner(dls, arch, loss_func, pretrained, cut, splitter, y_range, config, n_out, normalize, **kwargs)
    175     model = create_cnn_model(arch, n_out, ifnone(cut, meta['cut']), pretrained, y_range=y_range, **config)
    176     learn = Learner(dls, model, loss_func=loss_func, splitter=ifnone(splitter, meta['split']), **kwargs)
--> 177     if pretrained: learn.freeze()
    178     return learn
    179 

~/miniconda3/lib/python3.8/site-packages/fastai/learner.py in freeze(self)
    513 
    514 @patch
--> 515 def freeze(self:Learner): self.freeze_to(-1)
    516 
    517 @patch

~/miniconda3/lib/python3.8/site-packages/fastai/learner.py in freeze_to(self, n)
    508 @patch
    509 def freeze_to(self:Learner, n):
--> 510     if self.opt is None: self.create_opt()
    511     self.opt.freeze_to(n)
    512     self.opt.clear_state()

~/miniconda3/lib/python3.8/site-packages/fastai/learner.py in create_opt(self)
    139     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)
    140     def create_opt(self):
--> 141         self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
    142         if not self.wd_bn_bias:
    143             for p in self._bn_bias_state(True ): p['do_wd'] = False

<ipython-input-133-20a3ebb82957> in SGD_opt(params, **kwargs)
      1 from fastai.vision.all import *
      2 
----> 3 def SGD_opt(params, **kwargs): return OptimWrapper(torch.optim.SGD(params, **kwargs))
      4 
      5 path = untar_data(URLs.PETS)/'images'

~/miniconda3/lib/python3.8/site-packages/torch/optim/sgd.py in __init__(self, params, lr, momentum, dampening, weight_decay, nesterov)
     66         if nesterov and (momentum <= 0 or dampening != 0):
     67             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
---> 68         super(SGD, self).__init__(params, defaults)
     69 
     70     def __setstate__(self, state):

~/miniconda3/lib/python3.8/site-packages/torch/optim/optimizer.py in __init__(self, params, defaults)
     49 
     50         for param_group in param_groups:
---> 51             self.add_param_group(param_group)
     52 
     53     def __getstate__(self):

~/miniconda3/lib/python3.8/site-packages/torch/optim/optimizer.py in add_param_group(self, param_group)
    208         for param in param_group['params']:
    209             if not isinstance(param, torch.Tensor):
--> 210                 raise TypeError("optimizer can only optimize Tensors, "
    211                                 "but one of the params is " + torch.typename(param))
    212             if not param.is_leaf:

TypeError: optimizer can only optimize Tensors, but one of the params is list
```

The error is due to the fact that pytorch optimizers want the param list to be of the format `list(dict(params='model_parameters'))`. In case of fastai this is `list(list(params='model_parameters'))`.

It was also verified that the error does not occur when discriminative learning is not used.

One possible solution to the problem is to update the splitter to output `list(dict)` as shown below
```python
def splitter(m):
    ps = L(m[0][:3], m[0][3:], m[1:]).map(params)
    param_list = []
    for p in ps: param_list.append(dict(params=p))
    return p
```

I don't know if this is the best solution but as of now it get the work done.

Working code
```python
from fastai.vision.all import *

def SGD_opt(params, **kwargs): return OptimWrapper(torch.optim.SGD(params, **kwargs))

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate, opt_func=SGD_opt, splitter=splitter)
learn.fit_one_cycle(1)
```
Maybe the splitter code can be made part of `OptimWrapper` but I am not sure.

### Bugs Squashed

- Updated to support adding transforms to multiple dataloaders ([#3268](https://github.com/fastai/fastai/pull/3268)), thanks to [@marii-moe](https://github.com/marii-moe)
  - Fixes the issue here: https://forums.fast.ai/t/performance-degradation-between-fastai-2-2-5-and-2-2-7/86069

We add `add_tfms` in order to support adding the tfms to both dataloaders. Why?

Previously dls.train.after_batch.fs == dls.valid.after_batch.fs, so adding a transform to the training dataloader automatically added it to the validation dataloader. Removing that requirement caused validation to break, so adding back that functionality here. (moving my analysis from discord into the forum thread now)

- BrokenProcessPool in fastai.vision.utils.download_images() on Windows ([#3196](https://github.com/fastai/fastai/issues/3196))
  - Please confirm you have the latest versions of fastai, fastcore, and nbdev prior to reporting a bug (delete one): YES

**Describe the bug**
<details><summary>When calling `fastai.vision.utils.download_images()` with a URL list and destination Path, some students of mine* who were running on Windows received a BrokenProcessPool exception (click to expand):</summary>

```
~\Documents\hands on ai\Notebook 7\u7_utils.py in download_all_images(path)
    105         dest = path/classname
    106         dest.mkdir(exist_ok=True)
--> 107         download_images(dest, url_file=csv, max_pics=200)
    108     print("Done.")
    109 

~\miniconda3\envs\handsoai\lib\site-packages\fastai\vision\utils.py in download_images(dest, url_file, urls, max_pics, n_workers, timeout, preserve_filename)
     35     dest = Path(dest)
     36     dest.mkdir(exist_ok=True)
---> 37     parallel(partial(_download_image_inner, dest, timeout=timeout, preserve_filename=preserve_filename),
     38              list(enumerate(urls)), n_workers=n_workers)
     39 

~\miniconda3\envs\handsoai\lib\site-packages\fastcore\parallel.py in parallel(f, items, n_workers, total, progress, pause, threadpool, timeout, chunksize, *args, **kwargs)
    104             if total is None: total = len(items)
    105             r = progress_bar(r, total=total, leave=False)
--> 106         return L(r)
    107 
    108 # Cell

~\miniconda3\envs\handsoai\lib\site-packages\fastcore\foundation.py in __call__(cls, x, *args, **kwargs)
     95     def __call__(cls, x=None, *args, **kwargs):
     96         if not args and not kwargs and x is not None and isinstance(x,cls): return x
---> 97         return super().__call__(x, *args, **kwargs)
     98 
     99 # Cell

~\miniconda3\envs\handsoai\lib\site-packages\fastcore\foundation.py in __init__(self, items, use_list, match, *rest)
    103     def __init__(self, items=None, *rest, use_list=False, match=None):
    104         if (use_list is not None) or not is_array(items):
--> 105             items = listify(items, *rest, use_list=use_list, match=match)
    106         super().__init__(items)
    107 

~\miniconda3\envs\handsoai\lib\site-packages\fastcore\basics.py in listify(o, use_list, match, *rest)
     54     elif isinstance(o, list): res = o
     55     elif isinstance(o, str) or is_array(o): res = [o]
---> 56     elif is_iter(o): res = list(o)
     57     else: res = [o]
     58     if match is not None:

~\miniconda3\envs\handsoai\lib\concurrent\futures\process.py in _chain_from_iterable_of_lists(iterable)
    482     careful not to keep references to yielded objects.
    483     """
--> 484     for element in iterable:
    485         element.reverse()
    486         while element:

~\miniconda3\envs\handsoai\lib\concurrent\futures\_base.py in result_iterator()
    609                     # Careful not to keep a reference to the popped future
    610                     if timeout is None:
--> 611                         yield fs.pop().result()
    612                     else:
    613                         yield fs.pop().result(end_time - time.monotonic())

~\miniconda3\envs\handsoai\lib\concurrent\futures\_base.py in result(self, timeout)
    437                 raise CancelledError()
    438             elif self._state == FINISHED:
--> 439                 return self.__get_result()
    440             else:
    441                 raise TimeoutError()

~\miniconda3\envs\handsoai\lib\concurrent\futures\_base.py in __get_result(self)
    386     def __get_result(self):
    387         if self._exception:
--> 388             raise self._exception
    389         else:
    390             return self._result

BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.
```
</details>

**To Reproduce**
Steps to reproduce the behavior:
1. Create a list of URLs, such as:
   ```python
   urls = ['https://www.bk.com/sites/default/files/03156-5%20EggnormousBurrito%20500x540_CR.png',
           'https://s3.eu-central-1.amazonaws.com/food-truck-data-eu-central-1/media/operator/app/a3c936cad578f6e01b6b54809fc10f1e.jpg',
           'http://www.noracooks.com/wp-content/uploads/2018/03/IMG_6057-2.jpg',
           'https://www.nahundfrisch.at/cache/images/recipes/resizecrop_mm_h370_w782_q100/0ae000020b896f9015780ed9fd21b168.jpg']
   ```
2. Import `download_images`:
   ```python
   from fastai.vision.utils import download_images
   ```
3. Run it:
   ```python
   from pathlib import Path
   download_images(Path('.'), urls=urls)
   ```

**Expected behavior**
The code should download all images, irrespective of the platform it is run on.

**Error with full stack trace**
At least for some versions of Windows, it fails with the stack trace given on top. As I lack access to Windows machines, I unfortunately don't know exactly what circumstances must be met, but at least 2 in 250 people were affected.

**Additional context**
Judging from https://stackoverflow.com/questions/43836876/processpoolexecutor-works-on-ubuntu-but-fails-with-brokenprocesspool-when-r and https://bugs.python.org/issue17874, this is a general problem with the `ProcessPoolExecutor` running from an interactive shell. Using a `ThreadPoolExecutor` would be an easy fix. As suggested in https://github.com/fastai/fastai/issues/978, this would require adding `threadpool=True` to the call in https://github.com/fastai/fastai/blob/0e01131/fastai/vision/utils.py#L37. (Sorry, I haven't looked into how to submit a PR to fastai, given that all the code seems to be generated from notebooks, so I'm just reporting the problem and a potential fix here.)
Pinging @jph00 since he wanted to be informed.

(\* Just if you're wondering, I've used the image downloader and cnn_learner in a university class to demonstrate what's possible with transfer learning and what isn't, and am of course giving proper credit to the fastai library and the corresponding [fastbook chapter](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)! Thank you for the great work!)


## 2.2.7

### Bugs Squashed

- Regression fix: Ensure `add_datepart` adds elapsed as numeric column ([#3230](https://github.com/fastai/fastai/pull/3230)), thanks to [@aberres](https://github.com/aberres)


## 2.2.6

### Bugs Squashed

- 2.2.5 was not released correctly - it was actually 2.2.3

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

