from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
torch.backends.cudnn.benchmark = True

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distrubuted training of CIFAR-10.
    Fastest speed is if you run as follows:
        python -m fastai.launch train_cifar.py"""
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    path = url2path(URLs.CIFAR)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    workers = min(16, num_cpus()//n_gpus)
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512//n_gpus,
                                      num_workers=workers).normalize(cifar_stats)
    learn = Learner(data, wrn_22(), metrics=accuracy)
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else: learn.to_distributed(gpu)
    learn.to_fp16()
    learn.fit_one_cycle(35, 3e-3, wd=0.4)

