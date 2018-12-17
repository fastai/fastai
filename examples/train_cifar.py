from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
torch.backends.cudnn.benchmark = True

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
):
    gpu = setup_distrib(gpu)
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    path = url2path(URLs.CIFAR)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512//n_gpu,
                                      num_workers=16//n_gpu).normalize(cifar_stats)
    learn = Learner(data, wrn_22(), metrics=accuracy)
    if gpu is not None: learn.distributed(gpu)
    learn.to_fp16()
    learn.fit_one_cycle(30, 3e-3, wd=0.4)

    #learn.fit_one_cycle(30, 3e-3, wd=0.4, pct_start=0.5)

