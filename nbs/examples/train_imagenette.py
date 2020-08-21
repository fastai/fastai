from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *
from fastai.callback.mixup import *
from fastscript import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dls(size, woof, bs, sh=0., workers=None):
    if size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else        : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    source = untar_data(path)
    if workers is None: workers = min(8, num_cpus())
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    if sh: batch_tfms.append(RandomErasing(p=0.3, max_count=3, sh=sh))
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label,
                       item_tfms=[RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)],
                       batch_tfms=batch_tfms)
    return dblock.dataloaders(source, path=source, bs=bs, num_workers=workers)

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    woof:  Param("Use imagewoof (otherwise imagenette)", int)=0,
    lr:    Param("Learning rate", float)=1e-2,
    size:  Param("Size (px: 128,192,256)", int)=128,
    sqrmom:Param("sqr_mom", float)=0.99,
    mom:   Param("Momentum", float)=0.9,
    eps:   Param("epsilon", float)=1e-6,
    epochs:Param("Number of epochs", int)=5,
    bs:    Param("Batch size", int)=64,
    mixup: Param("Mixup", float)=0.,
    opt:   Param("Optimizer (adam,rms,sgd,ranger)", str)='ranger',
    arch:  Param("Architecture", str)='xresnet50',
    sh:    Param("Random erase max proportion", float)=0.,
    sa:    Param("Self-attention", int)=0,
    sym:   Param("Symmetry for self-attention", int)=0,
    beta:  Param("SAdam softplus beta", float)=0.,
    act_fn:Param("Activation function", str)='Mish',
    fp16:  Param("Use mixed precision training", int)=0,
    pool:  Param("Pooling method", str)='AvgPool',
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    meta:  Param("Metadata (ignored)", str)=''
):
    "Training of Imagenette."

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
    if   opt=='adam'  : opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt=='rms'   : opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt=='sgd'   : opt_func = partial(SGD, mom=mom)
    elif opt=='ranger': opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    dls = get_dls(size, woof, bs, sh=sh)
    if not gpu: print(f'epochs: {epochs}; lr: {lr}; size: {size}; sqrmom: {sqrmom}; mom: {mom}; eps: {eps}')

    m,act_fn,pool = [globals()[o] for o in (arch,act_fn,pool)]

    for run in range(runs):
        print(f'Run: {run}')
        learn = Learner(dls, m(n_out=10, act_cls=act_fn, sa=sa, sym=sym, pool=pool), opt_func=opt_func, \
                metrics=[accuracy,top_k_accuracy], loss_func=LabelSmoothingCrossEntropy())
        if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        cbs = MixUp(mixup) if mixup else []

        n_gpu = torch.cuda.device_count()

        # The old way to use DataParallel, or DistributedDataParallel training:
        # if gpu is None and n_gpu: learn.to_parallel()
        # if num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

        # the context manager way of dp/ddp, both can handle single GPU base case.
        ctx = learn.parallel_ctx if gpu is None and n_gpu else learn.distrib_ctx

        with partial(ctx, gpu)(): # distributed traing requires "-m fastai.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            learn.fit_flat_cos(epochs, lr, wd=1e-2, cbs=cbs)
