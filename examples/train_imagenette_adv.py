from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time
from fastprogress import fastprogress
from fastai.general_optimizer import *

fastprogress.MAX_COLS = 80

def get_data(size, woof, bs, workers=None, use_lighting=False):
    path = Path('/mnt/fe2_disk')
    if   size<=128: path = path/('imagewoof-160' if woof else 'imagenette-160')
    elif size<=192: path = path/('imagewoof-320' if woof else 'imagenette-320')
    else          : path = path/('imagewoof'     if woof else 'imagenette'    )

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

def get_learner(lr, size, woof, bs, opt_func, gpu, epochs):
    data = get_data(size, woof, bs)
    bs_rat = bs/256
    lr *= bs_rat
    b_its = len(data.train_dl)

    ph1 = (TrainingPhase(epochs*0.5*b_its)
            .schedule_hp('lr', (lr/20,lr), anneal=annealing_cos)
            .schedule_hp('eps', (1e-4,1e-7), anneal=annealing_cos)
            )
    ph2 = (TrainingPhase(epochs*0.5*b_its)
            .schedule_hp('lr', (lr,lr/1e5), anneal=annealing_cos)
            .schedule_hp('eps', (1e-7,1e-7), anneal=annealing_cos)
            )
    learn = (Learner(data, models.xresnet50(),
             metrics=[accuracy,top_k_accuracy], wd=1e-3, opt_func=opt_func,
             bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy())
        .mixup(alpha=0.2)
        .to_fp16(dynamic=True)
    )
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu)

    gs = GeneralScheduler(learn, (ph1,ph2))
    learn.fit(epochs, lr=1, callbacks=gs)

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
    alpha = ((st['alpha_buffer'] + group['eps']).sqrt()
            ) if 'alpha_buffer' in st else mom.new_tensor(1.)
    clip = group['clip'] if 'clip' in group else 1e9
    alr = (st['alpha_buffer']).clamp_min_(clip)
    p.data.addcdiv_(-group['lr'], st['momentum_buffer'], alr)

@call_parse
def main(
        gpu:Param("GPU to run on", str)=None,
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size (px: 128,192,224)", int)=128,
        debias_mom: Param("Debias statistics", bool)=False,
        debias_sqr: Param("Debias statistics", bool)=False,
        opt: Param("Optimizer: 'adam','genopt','rms','sgd'", str)='genopt',
        alpha: Param("Alpha", float)=0.99,
        mom: Param("Momentum", float)=0.9,
        eps: Param("epsilon", float)=1e-7,
        decay: Param("Decay AvgStatistic (momentum)", bool)=False,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=128,
        ):
    """Distributed training of Imagenette.
    Fastest multi-gpu speed is if you run with: python -m fastai.launch"""

    # Pick one of these
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()

    moms = (mom,mom)
    stats = [
            AvgStatistic('momentum', mom,   scope=StatScope.Weight, decay=decay, debias=debias_mom),
            AvgSquare   ('alpha',    alpha, scope=StatScope.Weight, debias=debias_sqr),
            ConstStatistic('eps', eps), ConstStatistic('clip', 0.001),
            ]
    if   opt=='adam'  : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='rms'   : opt_func = partial(optim.RMSprop, alpha=alpha)
    elif opt=='genopt': opt_func = partial(GeneralOptimizer, on_step=on_step, stats=stats)
    else: raise Exception(f'unknown opt: {opt}')

    #opt_func = optim.SGD
    #learn = (cnn_learner(data, models.xresnet50, pretrained=False, concat_pool=False, lin_ftrs=[], split_on=bn_and_final,
    print(f'lr: {lr}; size: {size}; debias_mom: {debias_mom}; debias_sqr: {debias_sqr}; opt: {opt}; alpha: {alpha}; mom: {mom}; eps: {eps}; decay: {decay}')
    print('imagenette')
    get_learner(lr, size, False, bs, opt_func, gpu, epochs)
    gc.collect()

    print('imagewoof')
    get_learner(lr, size, True, bs, opt_func, gpu, epochs)

    #learn.recorder.plot_lr(show_moms=True)
    #learn.save('nette')

