from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_data(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=192: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

@call_parse
def main(
        woof: Param("Use imagewoof (otherwise imagenette)", bool)=False,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        alpha: Param("Alpha", float)=0.9,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-7,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=256,
        gpu:Param("GPU to run on", str)=None,
        ):
    "Distributed training of Imagenette."

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)

    print(f'lr: {lr}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    data = get_data(size, woof, bs)
    bs_rat = bs/256
    lr *= bs_rat

    learn = (Learner(data, models.xresnet50(),
             metrics=[accuracy,top_k_accuracy], wd=1e-3, opt_func=opt_func,
             bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy())
        .mixup(alpha=0.2)
        .to_fp16(dynamic=True)
    )
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.5)

