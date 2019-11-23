from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai2.vision.models.xresnet import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dbunch(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    source = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)
        
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files,
                       get_y=parent_label)

    return dblock.databunch(source, path=source, item_tfms=[RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)], bs=bs, num_workers=workers)

@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-6,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=256,
        mixup: Param("Mixup", float)=0.,
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50, presnet34, presnet50)", str)='xresnet50',
        dump: Param("Print model; don't train", int)=0,
        ):
    "Distributed training of Imagenette."

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(Adam, mom=mom, sqr_mom=alpha, eps=eps)
    elif opt=='rms'  : opt_func = partial(RMSprop, sqr_mom=alpha)
    elif opt=='sgd'  : opt_func = partial(SGD, mom=mom)

    dbunch = get_dbunch(size, woof, bs)
    bs_rat = bs/256
    if gpu is not None: bs_rat *= num_distrib()
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat

    m = globals()[arch]
    learn = (Learner(dbunch, m(c_out=10), opt_func=opt_func,
             metrics=[accuracy,top_k_accuracy],
             wd_bn_bias=False,
             loss_func = LabelSmoothingCrossEntropy())
            )
    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    learn = learn.to_fp16()
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    learn.fit_one_cycle(epochs, lr, div=10, pct_start=0.3, wd=1e-2)

