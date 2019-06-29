---
title: How to launch a distributed training
---

If you have multiple GPUs, the most reliable way to use all of them for training is to use the distributed package from pytorch. To help you, there is a distributed module in fastai that has helper functions to make it really easy.

## Prepare your script

Distributed training doesn't work in a notebook, so first, clean up your experiments notebook and prepare a script to run the training. For instance, here is a minimal script that trains a wide resnet on CIFAR10.

``` python
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22

path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=128).normalize(cifar_stats)
learn = Learner(data, wrn_22(), metrics=accuracy)
learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
```

## Add the distributed initialization

Your script is going to be executed in a different process that will each happen on a different GPU. To make this work properly, add the following introduction between your imports and the rest of your code.

``` python
from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
```

What we do here is that we import the necessary stuff from fastai (for later), we create an argument parser that will intercept an argument named `local_rank` (which will contain the name of the GPU to use), then we set our GPU accordingly. The last line is what pytorch needs to set things up properly and know that this process is part of a larger group.

## Make your learner distributed

You then have to add one thing to your learner before fitting it to tell it it's going to execute a distributed training:
``` python
learn = learn.to_distributed(args.local_rank)
```
This will add the additional callbacks that will make sure your model and your data loaders are properly setups.

Now you can save your scriptn here is what the full example looks like:

``` python
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=128).normalize(cifar_stats)
learn = Learner(data, wrn_22(), metrics=accuracy).to_distributed(args.local_rank)
learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
```

## Launch your training

In your terminal, type the following line (adapt `num_gpus` and `script_name` to the number of GPUs you want to use and your script name ending with .py).
```
python -m torch.distributed.launch --nproc_per_node={num_gpus} {script_name}
```

What will happen is that the same model will be copied on all your available GPUs. During training, the full dataset will randomly be split between the GPUs (that will change at each epoch). Each GPU will grab a batch (on that fractioned dataset), pass it through the model, compute the loss then back-propagate the gradients. Then they will share their results and average them, which means like your training is the equivalent of a training with a batch size of `batch_size x num_gpus` (where `batch_size` is what you used in your script). 

Since they all have the same gradients at this stage, they will al perform the same update, so the models will still be the same after this step. Then training continues with the next batch, until the number of desired iterations is done.
