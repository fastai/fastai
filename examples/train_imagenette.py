from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time

def get_data(path, size, bs, workers):
    tfms = ([
        flip_lr(p=0.5),
        brightness(change=(0.4,0.6)),
        contrast(scale=(0.7,1.3))
    ], [])
    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform(tfms, size=size).presize(size, scale=(0.35,1.0))
            .databunch(bs=bs, num_workers=workers).normalize(imagenet_stats))

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenette.
    Fastest multi-gpu speed is if you run with: python -m fastai.launch"""
    gpu = setup_distrib(gpu)
    n_gpus = rank_distrib() or 1

    path = untar_data(URLs.IMAGENETTE_160)
    tot_epochs,size,lr = 40,128,0.6
    bs = 256//n_gpus

    workers = min(12, num_cpus()//n_gpus)
    data = get_data(path, size, bs, workers)
    opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=0.3)
    learn = Learner(data, models.xresnet50(), metrics=[accuracy,top_k_accuracy], wd=1e-3,
        opt_func=opt_func, bn_wd=False, true_wd=True,
        loss_func = LabelSmoothingCrossEntropy()).mixup(alpha=0.2)
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else:           learn.distributed(gpu)
    learn.to_fp16(dynamic=True)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    tot_bs = bs*n_gpus
    bs_rat = tot_bs/256
    lr *= bs_rat
    learn.fit_one_cycle(tot_epochs, lr, div_factor=20, pct_start=0.7, moms=(0.9,0.9))
    learn.save('nette')

