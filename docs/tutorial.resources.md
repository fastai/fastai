---
title: Deep Learning on a Shoestring
---

## Introduction

Most machine learning tasks are very compute intensive and often require a lot of expensive hardware, which one never has enough of, since it's almost always the case that one could do better if only there were more hardware. GPU RAM in a particular is the main resource that is always lacking. This tutorial will focus on various topics that will explain how you could accomplish impressive feats without needing spending more money on hardware acquisition.

**Note: this tutorial is a work in progress and needs more content and polishing, yet, it's already quite useful. It will get improved over time and contributions are welcome too.**

## Lean and mean notebook coding

A typical machine learning notebook consumes more and more GPU RAM, and often in order to complete training one needs to restart the notebook kernel multiple times, then carefully rerun some cells, jumping over many other cells, then running some more cells, hitting the memory limit, and repeating this process again, this time carefully skipping yet other sets of cells and running yet another few cells until again the dreadful OOM is encountered.

So, the first thing to understand is that as you do the training the GPU RAM consumption gradually increases. But at any stage you can reclaim most of the GPU RAM used so far, by saving your model under training, freeing all other no longer needed objects and then reloading the saved model.

This is usually not a problem in a normal programming environment, where well-behaved functions cleanup after themselves and leak no memory, but in jupyter notebook the mindset is different - there is just one function spread out to many cells and each cell may consume memory, but since its variables never go out of scope the memory is tied up until the notebook kernel is restarted.

So a typical CNN notebook goes as following:

```
# imports
import ...

# init main objects
model = ... # custom or pretrained
data = ...  # create databunch
learn = cnn_learner(data, model, metrics)

# train
learn.fit_one_cycle(epochs)

# finetune
learn.unfreeze
learn.fit_one_cycle(epochs)

# predict
preds,y = learn.get_preds(...)
```

Typically if you picked the batch size right and your model is not too deep and dense, you may be able to complete this whole notebook on, say, 8GB of GPU RAM. If the model you picked is memory intensive, you might successfully complete the training of the NN head added by fastai (the part before unfreeze), but will be unable to finetune because its memory requirements are much bigger. Finally, you might be able to have just enough memory to finetune, but, alas, no memory will be left for the prediction phase (especially if it's a complex prediction).

But, of course, this was a very basic example. A more advanced approach will, for example, start with the basic training, followed by finetuning stage, then will change the size of the input images and repeat head training and finetuning, and then again. And there are many other stages that can be added.

The key is to be able to reclaim GPU and general RAM between all those stages so that a given notebook could do many many things without needing to be restarted.

In a normal function, once its finished, all of its local variables will get destroyed, which should reclaim any memory used by those. If a variable is an object that has circular references, it won't release its memory until a scheduled call of `gc.collect` will arrive and do the release.

In the case of the notebooks there is no automatic variable destruction, since the whole notebook is just one single scope and only shutting down the notebook kernel will free the memory automatically. And then the `Learner` object is a very complex beast that can't avoid circular references and therefore even if you do `del learn` it won't free its resources automatically. We can't afford waiting for `gc.collect` to arrive some time in the future, as we need the RAM now, or we won't be able to continue. Therefore we have to manually call `gc.collect`.

Therefore if we want to free memory half-way through the notebook, the awkward way to do so is:

```
del learn
gc.collect()
learn = ... # reconstruct learn
```

There is also [ipyexperiments](https://github.com/stas00/ipyexperiments/) that was initially created to solve this exact problem - to automate this process of auto-deleting variables by creating artificial scopes (the module itself since evolved to do much more).

footnote: you may also want to call `torch.cuda.empty_cache` if you actually want to see the memory freed with `nvidia-smi` - this is due to pytorch caching. The memory is freed, but you can't tell that from `nvidia-smi`. You can if you use [pytorch memory allocation functions](https://pytorch.org/docs/stable/notes/cuda.html#memory-management), and it's another function to call) - `ipyexperiments` will do it automatically for you.

But the annoying and error-prone part is that either way you have to reconstruct the `Learner` object and potentially other intermediary objects.

Let's welcome a new function: `learn.purge` that solves this problem transparently. The last snippet of code is thus gets replaced with just:

```
learn.purge()
```
which removes any of the `Learner` guts that are no longer needed and reloads the model on GPU, which also helps to reduce [memory fragmentation](/dev/gpu.html#gpu-ram-fragmentation). Therefore, whenever you need the no longer memory purges, this is the way to do it.

Furthermore, the purging functionality is included in `learn.load` if you pass `purge=True`.

So instead of needing to do:

```
learn.fit_one_cycle(epochs)

learn.save('saved')
learn.load('saved')
learn.purge()
```

the call to `learn.purge` is not needed.

You don't need to inject `learn.purge` between training cycles of the same setup:
```
learn.fit_one_cycle(epochs=10)
learn.fit_one_cycle(epochs=10)
```
The subsequent invocations of the training function do not consume more GPU RAM. Remember, when you train you just change the numbers in the nodes, but all the memory that is required for those numbers has already been allocated.



### Optimizer nuances

The `learn` object can be reset between fit cycles to save memory. But you need to know when it's the right and practical thing to do. For example in a situation like this:

```
learn.fit_one_cycle(epochs=10)
learn.fit_one_cycle(epochs=10)
```

purging the `learn` object between two `fit` calls will make no difference to GPU RAM consumption. But if you had other things happening in between, such as freeze/unfreeze or image size change, you should be able to recover quite a lot of GPU RAM.

As explained in the previous section `learn.load` internally calls `learn.purge`, but by default it doesn't clear `learn.opt` (the optimizer state). It's a safe default, because, some models like GANs break if `learn.opt` gets cleared between `fit` cycles.

However, `learn.purge` does clear `learn.opt` by default.

Therefore, if your model is sensitive to clearing `learn.opt`, you should be using one of these 2 ways:

1.
   ```
   learn.load(name) # which calls learn.load(name, with_opt=True)
   ```
2.
   ```
   learn.purge(clear_opt=False)
   ```

either of which will reset everything in `learn`, except `learn.opt`.

If your model is not sensitive to `learn.opt` resetting between fit cycles, and you
want to reclaim more GPU RAM, you can clear `learn.opt` via one of the following 3 ways:

1.
   ```
   learn.load('...', with_opt=False)
   ```
2.
   ```
   learn.purge()  # which calls learn.purge(clear_opt=True)
   ```
3.
   ```
   learn.opt.clear()
   ```

### Learner release

If you no longer need the `learn` object, you can release the memory it is consuming by calling:

```
del learn
gc.collect()
```
`learn` is a very complex object with multiple sub-objects, with unavoidable circular references, so `del learn` won't free the memory until `gc.collect` arrives some time in the future, so since we need the memory now, we call it directly.

If the above code looks a bit of an eye-sore, do the following instead:

```
learn.destroy()
```
`destroy` will release all the memory consuming parts of it, while leaving an empty shell - the object will still be there, but you won't be able to do anything with it.


### Inference

For inference we only need the saved model and the data to predict on, and nothing else that was used during the training. So to use even less memory (general RAM this time), the lean approach is to `learn.export` and `learn.destroy` at the end of the training, and then to `load_learner` before the prediction stage is started.

```
# end of a potential training scenario
learn.unfreeze()
learn.fit_one_cycle(epochs)
learn.freeze()
learn.export(destroy=True) # or learn.export() + learn.destroy()

# beginning of inference
learn = load_learner(path, test=ImageList.from_folder(path/'test'))
preds = learn.get_preds(ds_type=DatasetType.Test)
```


### Conclusion

So, to conclude, when you finished your `learn.fit` cycles and you are changing to a different image size, or you unfreeze, or you do anything else that no longer requires previous structures on GPU, you either call `learn.purge` or `learn.load('saved_name')` and you should have most of your GPU RAM back as it was where you have just started, plus the allocated memory for the model. That's of course, if you haven't created some other variables that hold some GPU RAM tied up.

And for inferences the  `learn.export`, `learn.destroy` and `load_learner` sequence will require even less RAM.

Therefore now you should be able to do a ton of things without needing to restart your notebook.

### Local variables

python releases memory when variable's reference count goes to 0 (and in special cases where circular references are referring to each other but have no "real" variables that refer to them). Sometimes you could be trying hard to free some memory, including calling `learn.destroy` and some memory refuses to go. That usually means that you have some local variables that hold memory. Here is an example:

```
class FeatureLoss(nn.Module): [...]
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
learn = unet_learner(data, arch, loss_func=feat_loss, ...)
[...]
learn.destroy()
```
In this example, `learn.destroy` won't be able to release a very large chunk of GPU RAM, because the local variable `feat_loss` happens to hold a reference to the loss function which, while originally takes up almost no memory, tends to grow pretty large during `unet` training. Once you delete it, python will be able to release that memory. So it's always best to get rid of intermediate variables of this kind (not simple values like `bs=64`) as soon as you don't need them anymore. Therefore, in this case the more efficient code would be:

```
class FeatureLoss(nn.Module): [...]
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
learn = unet_learner(data, arch, loss_func=feat_loss, ...)
del feat_loss
[...]
learn.destroy()
```


## CUDA out of memory

One of the main culprits leading to a need to restart the notebook is when the notebook runs out of memory with the known to all `CUDA out of memory` (OOM) exception. This is covered in this [section](/troubleshoot.html#cuda-out-of-memory-exception).

And also you need to know about the current bug in ipython that may prevent you from being able to continue to use the notebook on OOM. This problem is mainly taken care of automatically in fastai, and is explained in details [here](/troubleshoot.html#memory-leakage-on-exception).


## GPU Memory Usage Anatomy

1. About 0.5GB per process is used by CUDA context, see [Unusable GPU RAM per process](/dev/gpu.html#unusable-gpu-ram-per-process). This memory is consumed during the first call to `.cuda`, when the first tensor is moved to GPU. You can test your card with:

   ```
   python -c 'from fastai.utils.mem import *; b=gpu_mem_get_used_no_cache(); preload_pytorch(); print(gpu_mem_get_used_no_cache()-b);'
   ```

2. A pre-trained model consumes a little bit of GPU RAM. While there are many different models out there, which may vary wildly in size, a freshly loaded pre-trained model like resnet typically consumes a few hundred MBs of GPU RAM.

3. Model training is where the bulk of GPU RAM is being consumed. When the very first batch of the very first epoch goes through the model, the GPU RAM usage spikes because it needs to set things up and a lot more temporary memory is used than on subsequent batches. However the pytorch allocator is very efficient and if there is little GPU RAM available, the spike will be minimal. From batch 2 and onwards and for all the following epochs of the same `fit` call the memory consumption is constant. Thus if the first few seconds of `fit` were successful, it will run to its completion.

   Tip: if you're tuning hyperparameters to fit into your card's GPU RAM, it's enough to run just one epoch of each `fit` call, so that you can quickly choose `bs` and other parameters to fit the available RAM. After this is done, increase the number of epochs in `fit` calls to get the real training going.

If you'd like to get a sense of how much memory each stage uses (bypassing pytorch caching, which you can't with `nvidia-smi`), here are the tools to use:

* For per-notebook and per-cell: [ipyexperiments](https://github.com/stas00/ipyexperiments/).
* For per-epoch use: [`PeakMemMetric`](/callbacks.mem.html#PeakMemMetric).
* For even more fine-grained profiling:  [GPUMemTrace](/utils.mem.html#GPUMemTrace).


**Real memory usage**

* pytorch caches memory through its memory allocator, so you can't use tools like `nvidia-smi` to see how much real memory is available. So you either need to use pytorch's memory management functions to get that information or if you want to rely on `nvidia-smi` you have to flush the cache. Refer to this [document](/dev/gpu.html#cached-memory) for details.


## TODO/Help Wanted

This tutorial is a work in progress, some areas partially covered, some not at all. So if you have the know how or see something is missing please submit a PR.

Here is a wish list:

* [`torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html) can be used to use less GPU RAM by re-computing gradients. Here is a good in-depth [article](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) explaining this feature in tensorflow, and [another one](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) that talks about the theory. Here is the [notebook](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb) from the person who contributed this feature to pytorch (still uses pre pytorch-0.4 syntax). We need pytorch/fastai examples. Contributions are welcome.
