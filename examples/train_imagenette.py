from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time
from fastai.general_optimizer import *

def get_data(path, size, bs, workers=None, use_lighting=False):
    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    tfms = [flip_lr(p=0.5)]
    if use_lighting:
        tfms += [brightness(change=(0.4,0.6)), contrast(scale=(0.7,1.3))]
    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform((tfms, []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

def bn_and_final(m):
    ll = flatten_model(m)
    last_lin = next(o for o in reversed(ll) if isinstance(o, bias_types))
    idx = [i for i,o in enumerate(ll) if
           (i>50 and isinstance(o, bn_types)) or o==last_lin]
    l1 = [o for i,o in enumerate(ll) if i not in idx]
    l2 = [ll[i] for i in idx]
    return split_model(splits=[l1,l2])

def on_step(self, p, group, group_idx):
    st = self.state[p]
    mom = st['momentum_buffer']
    alpha = (st['alpha_buffer'].sqrt()+1e-7
            ) if 'alpha_buffer' in st else mom.new_tensor(1.)
    p.data.addcdiv_(-group['lr'], mom, alpha)

@call_parse
def main(
        woof:Param("Use woof", bool)=False,
        gpu:Param("GPU to run on", str)=None,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        debias: Param("Debias statistics", bool)=False,
        decay: Param("Decay AvgStatistic (momentum)", bool)=False,
        ):
    """Distributed training of Imagenette.
    Fastest multi-gpu speed is if you run with: python -m fastai.launch"""
    bs,tot_epochs = 256,5

    # Pick one of these
    if   size<=128: path = untar_data(URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160)
    elif size<=192: path = untar_data(URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320)
    else          : path = untar_data(URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE    )

    gpu = setup_distrib(gpu)

    data = get_data(path, size, bs)
    #opt_func = partial(optim.Adam, betas=(0.9,0.9), eps=1e-7)
    #opt_func = partial(optim.RMSprop, alpha=0.9)
    #opt_func = optim.SGD
    stats = [AvgStatistic('momentum', 0.9, scope=StatScope.Weight, decay=decay, debias=debias),
             AvgSquare   ('alpha',    0.9, scope=StatScope.Weight, debias=debias)]
    opt_func = partial(GeneralOptimizer, on_step=on_step, stats=stats)
    #learn = (cnn_learner(data, models.xresnet50, pretrained=False, concat_pool=False, lin_ftrs=[], split_on=bn_and_final,
    learn = (Learner(data, models.xresnet50(),
             metrics=[accuracy,top_k_accuracy], wd=1e-3, opt_func=opt_func,
             bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy())
        .mixup(alpha=0.2)
        .to_fp16(dynamic=True)
        #.split(bn_and_final)
    )
    learn.callback_fns += [
        partial(TrackEpochCallback),
        partial(SaveModelCallback, every='epoch', name='model')
    ]
    if gpu is None: learn.to_parallel()
    else:           learn.to_distributed(gpu)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    bs_rat = bs/256
    lr *= bs_rat
    learn.fit_one_cycle(tot_epochs, lr, div_factor=20, pct_start=0.5, moms=(0.9,0.9))
    #learn.recorder.plot_lr(show_moms=True)
    #learn.save('nette')

