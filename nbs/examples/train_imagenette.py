from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *
from fastai.callback.mixup import *
from fastcore.script import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
def pr(s):
    if rank_distrib()==0: print(s)

def get_dls(size, woof, pct_noise, bs, sh=0., workers=None):
    assert pct_noise in [0,5,50], '`pct_noise` must be 0,5 or 50.'
    if size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else        : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    source = untar_data(path)
    workers = ifnone(workers,min(8,num_cpus()))
    blocks=(ImageBlock, CategoryBlock)
    tfms = [RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)]
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    if sh: batch_tfms.append(RandomErasing(p=0.3, max_count=3, sh=sh))
    
    if pct_noise > 0:
        csv_file = 'noisy_imagewoof.csv' if woof else 'noisy_imagenette.csv'
        inp = pd.read_csv(source/csv_file)
        dblock = DataBlock(blocks=blocks,
                   splitter=ColSplitter(),
                   get_x=ColReader('path', pref=source), 
                   get_y=ColReader(f'noisy_labels_{pct_noise}'),
                   item_tfms=tfms,
                   batch_tfms=batch_tfms)
    else:
        inp = source
        dblock = DataBlock(blocks=blocks,
                   splitter=GrandparentSplitter(valid_name='val'),
                   get_items=get_image_files, get_y=parent_label,
                   item_tfms=tfms,
                   batch_tfms=batch_tfms)
    
    return dblock.dataloaders(inp, path=source, bs=bs, num_workers=workers)

@call_parse
def main(
    woof:  Param("Use imagewoof (otherwise imagenette)", int)=0,
    pct_noise:Param("Percentage of noise in training set (1,5,25,50%)", int)=0,
    lr:    Param("Learning rate", float)=1e-2,
    size:  Param("Size (px: 128,192,256)", int)=128,
    sqrmom:Param("sqr_mom", float)=0.99,
    mom:   Param("Momentum", float)=0.9,
    eps:   Param("Epsilon", float)=1e-6,
    wd:    Param("Weight decay", float)=1e-2,
    epochs:Param("Number of epochs", int)=5,
    bs:    Param("Batch size", int)=64,
    mixup: Param("Mixup", float)=0.,
    opt:   Param("Optimizer (adam,rms,sgd,ranger)", str)='ranger',
    arch:  Param("Architecture", str)='xresnet50',
    sh:    Param("Random erase max proportion", float)=0.,
    sa:    Param("Self-attention", store_true)=False,
    sym:   Param("Symmetry for self-attention", int)=0,
    beta:  Param("SAdam softplus beta", float)=0.,
    act_fn:Param("Activation function", str)='Mish',
    fp16:  Param("Use mixed precision training", store_true)=False,
    pool:  Param("Pooling method", str)='AvgPool',
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    meta:  Param("Metadata (ignored)", str)='',
    workers:   Param("Number of workers", int)=None,
):
    "Training of Imagenette. Call with `python -m fastai.launch` for distributed training"
    if   opt=='adam'  : opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt=='rms'   : opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt=='sgd'   : opt_func = partial(SGD, mom=mom)
    elif opt=='ranger': opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    dls = rank0_first(get_dls, size, woof, pct_noise, bs, sh=sh, workers=workers)
    pr(f'epochs: {epochs}; lr: {lr}; size: {size}; sqrmom: {sqrmom}; mom: {mom}; eps: {eps}')
    m,act_fn,pool = [globals()[o] for o in (arch,act_fn,pool)]

    for run in range(runs):
        pr(f'Run: {run}')
        learn = Learner(dls, m(n_out=10, act_cls=act_fn, sa=sa, sym=sym, pool=pool), opt_func=opt_func, \
                metrics=[accuracy,top_k_accuracy], loss_func=LabelSmoothingCrossEntropy())
        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_native_fp16()
        cbs = MixUp(mixup) if mixup else []
        n_gpu = torch.cuda.device_count()
        # Both context managers work fine for single GPU too
        ctx = learn.distrib_ctx if num_distrib() and n_gpu else learn.parallel_ctx
        with ctx(): learn.fit_flat_cos(epochs, lr, wd=wd, cbs=cbs)

