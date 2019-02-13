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
learn = create_cnn(data, model, metrics)

# train
learn.fit_one_cycle(epochs)

# finetune
learn.unfreeze
learn.fit_one_cycle(epochs)

# predict
preds,y = learn.get_preds(...)
```

Typically if you picked the batch size right and your model is not too deep and dense, you may be able to complete this whole notebook on say 8GB of GPU RAM. If the model you picked is memory intensive, you might successfully complete the training of the NN head added by fastai (the part before unfreeze), but will be unable to finetune because its memory requirements are much bigger. Finally, you might be able to have just enough memory to finetune, but alas no memory will be left for the prediction phase (especially if it's a complex prediction).

But, of course, this was a very basic example. A more advanced approach will, for example, start with the basic training, followed by finetuning stage, then will change the size of the input images and repeat head training and finetuning, and then again. And there are many other stages that can be added.

The key is to be able to reclaim GPU and general RAM between all those stages so that a given notebook could do many many things without needing to be restarted.

In a normal function, once its finished, all of its local variables will get destroyed, which should reclaim any memory used by those. If a variable is an object that has circular references, it won't release its memory until a scheduled call of `gc.collect()` will arrive and do the release.

In the case of the notebooks there is no automatic variable destruction, since the whole notebook is just one single scope and only shutting down the notebook kernel will free the memory automatically. And then the `Learner` object is a very complex beast that can't avoid circular references and therefore even if you do `del learn` it won't free its resources automatically. We can't afford waiting for `gc.collect()` to arrive some time in the future, as we need the RAM now, or we won't be able to continue. Therefore we have to manually call `gc.collect`.

Therefore if we want to free memory half-way through the notebook, the awkward way to do so is:

```
del learn
gc.collect()
learn = ... # reconstruct learn
```

There is also [ipyexperiments](https://github.com/stas00/ipyexperiments/) that was initially created to solve this exact problem - to automate this process of auto-deleting variables by creating artificial scopes (the module itself since evolved to do much more).

footnote: you may also want to call `torch.cuda.empty_cache()` if you actually want to see the memory freed with `nvidia-smi` - this is due to pytorch caching. The memory is freed, but you can't tell that from `nvidia-smi`. You can if you use [pytorch memory allocation functions](https://pytorch.org/docs/stable/notes/cuda.html#memory-management), and it's another function to call) - `ipyexperiments` will do it automatically for you.

But the annoying and error-prone part is that either way you have to reconstruct the `Learner` object and potentially other intermediary objects.

Let's welcome a new function: `learn.purge()` that solves this problem transparently. The last snippet of code is thus gets replaced with just:

```
learn.purge()
```
which removes any of the `Learner` guts that are no longer needed and reloads the model on GPU, which also helps to reduce [memory fragmentation](https://docs.fast.ai/dev/gpu.html#gpu-ram-fragmentation). Therefore, whenever you need the no longer memory purges, this is the way to do it.

Furthermore, the purging functionality is included in `learn.load()` and is performed by default. You can override the default behavior of it not to purge with `purge=False` argument).

So instead of needing to do:

```
learn.fit_one_cycle(epochs)

learn.save('saved')

learn.purge()
learn.load('saved')
```

the call to `learn.purge()` is not needed.



### Inference

For inference we only need the saved model and the data to predict on, and nothing else that was used during the training. So to use even less memory (general RAM this time), the lean approach is to `learn.export()` and `learn.purge()` at the end of the training, and then to `load_learner()` before the prediction stage is started.

```
# end of training
learn.fit_one_cycle(epochs)
learn.freeze()
learn.export()
learn.purge()

# beginning of inferences
learn = load_learner(path, test=ImageItemList.from_folder(path/'test'))
preds = learn.get_preds(ds_type=DatasetType.Test)
```


### Conclusion

So, to conclude, when you finished your `learn.fit()` cycles and you are changing to a different image size, or you unfreeze, or you do anything else that no longer requires previous structures on GPU, you either call `learn.purge()` or `learn.load('saved_name')` and you should have most of your GPU RAM back as it was where you have just started, plus the allocated memory for the model. That's of course, if you haven't created some other variables that hold some GPU RAM tied up.

And for inferences the  `learn.export()`, `learn.purge()` and `load_learner()` sequence will require even less RAM.

Therefore now you should be able to do a ton of things without needing to restart your notebook.


## CUDA out of memory

One of the main culprits leading to a need to restart the notebook is when the notebook runs out of memory with the known to all `CUDA out of memory` (OOM) exception. This is covered in this [section](https://docs.fast.ai/troubleshoot.html#cuda-out-of-memory-exception).

And also you need to know about the current bug in ipython that may prevent you from being able to continue to use the notebook on OOM. This problem is mainly taken care of automatically in fastai, and is explained in details [here](https://docs.fast.ai/troubleshoot.html#memory-leakage-on-exception).


## GPU Memory Usage Anatomy

1. About 0.5GB per process is used by CUDA context, see [Unusable GPU RAM per process](https://docs.fast.ai/dev/gpu.html#unusable-gpu-ram-per-process). This memory is consumed during the first call to `.cuda()`, when the first tensor is moved to GPU.

2. A pre-trained model consumes a little bit of GPU RAM. While there are many different models out there, which may vary wildly in size, a freshly loaded pre-trained model like resnet typically consumes a few hundred MBs of GPU RAM.

3. Model training is where the bulk of GPU RAM is being consumed. When the very first batch of the very first epoch goes through the model, the GPU RAM usage spikes because it needs to set things up and a lot more temporary memory is used. However the pytorch allocator is very efficient and if there is little GPU RAM available, the spike will be minimal. From batch 2 onwards and for all the following epoch of the same training the memory consumption would be constant. Thus if the first few seconds of training were successful, the rest of the training should complete too.

If you'd like to get a sense of how much memory each stage uses (bypassing pytorch caching, which you can't with `nvidia-smi`), here are the tools to use:

* For per-notebook and per-cell: [ipyexperiments](https://github.com/stas00/ipyexperiments/).
* For per-epoch use: [`PeakMemMetric`](/callbacks.mem.html#PeakMemMetric).
* For even more fine-grained profiling:  [GPUMemTrace](https://docs.fast.ai/utils.mem.html#GPUMemTrace).


**Real memory usage**

* pytorch caches memory through its memory allocator, so you can't use tools like `nvidia-smi` to see how much real memory is available. So you either need to use pytorch's memory management functions to get that information or if you want to rely on `nvidia-smi` you have to flush the cache. Refer to this [document](https://docs.fast.ai/dev/gpu.html#cached-memory) for details.
