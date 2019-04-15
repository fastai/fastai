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
    train = ImageList.from_csv(path, 'train.csv')
    valid = ImageList.from_csv(path, 'valid.csv')
    lls = ItemLists(path, train, valid).label_from_df().transform(
            tfms, size=size).presize(size, scale=(0.25, 1.0))
    return lls.databunch(bs=bs, num_workers=workers).normalize(imagenet_stats)

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenet. Fastest speed is if you run with: python -m fastai.launch"""
    path = Path('/mnt/fe2_disk/')
    tot_epochs,size,bs,lr = 60,224,256,3e-1
    dirname = 'imagenet'

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    n_gpus = num_distrib() or 1
    workers = min(12, num_cpus()//n_gpus)
    data = get_data(path/dirname, size, bs, workers)
    b_its = len(data.train_dl)//n_gpus

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    tot_bs = bs*n_gpus
    bs_rat = tot_bs/256
    lr *= bs_rat

    ph1 = (TrainingPhase(tot_epochs*0.10*b_its)
            .schedule_hp('lr', (lr/10,lr),  anneal=annealing_cos))
    ph2 = (TrainingPhase(tot_epochs*0.90*b_its)
            .schedule_hp('lr', (lr,lr/1e5), anneal=annealing_cos))
    opt_func = partial(optim.Adam, eps=0.1, betas=(0.9,0.99))
    learn = Learner(data, models.xresnet50(), metrics=[accuracy,top_k_accuracy], wd=1e-3,
        opt_func=opt_func, bn_wd=False, true_wd=True,
        loss_func = LabelSmoothingCrossEntropy()).mixup(alpha=0.2)

    learn.callback_fns += [
        partial(GeneralScheduler, phases=(ph1,ph2)),
        partial(SaveModelCallback, every='epoch', name='model')
    ]
    learn.split(lambda m: (children(m)[-2],))
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else:           learn.to_distributed(gpu)
    learn.to_fp16(dynamic=True)

    learn.fit(tot_epochs, 1)
    if rank_distrib(): time.sleep(1)
    learn.save('done')

