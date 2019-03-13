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
    train = ImageList.from_csv(path, 'train.csv')#.use_partial_data(0.001)
    valid = ImageList.from_csv(path, 'valid.csv')#.use_partial_data()
    lls = ItemLists(path, train, valid).label_from_df().transform(
            tfms, size=size).presize(size, scale=(0.35, 1.0))
    return lls.databunch(bs=bs, num_workers=workers).normalize(imagenet_stats)

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenet. Fastest speed is if you run with: python -m fastai.launch"""
    path = Path('/mnt/fe2_disk/')
    tot_epochs,size,bs,lr = 90,224,256,0.6
    dirname = 'imagenet'

    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()

    n_gpus = num_distrib() or 1
    workers = min(12, num_cpus()//n_gpus)
    data = get_data(path/dirname, size, bs, workers)

    opt_func = partial(optim.SGD, momentum=0.9)
    learn = Learner(data, models.xresnet50(), metrics=[accuracy,top_k_accuracy], wd=1e-5,
        opt_func=opt_func, bn_wd=False, true_wd=False,
        loss_func = LabelSmoothingCrossEntropy()).mixup(alpha=0.2)
    learn.callback_fns += [
        partial(TrackEpochCallback),
        partial(SaveModelCallback, every='epoch', name='model')
    ]
    learn.split(lambda m: (children(m)[-2],))
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else:           learn.to_distributed(gpu)
    learn.to_fp16(dynamic=True)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    tot_bs = bs*n_gpus
    bs_rat = tot_bs/256
    lr *= bs_rat
    learn.fit_one_cycle(tot_epochs, lr, div_factor=5, moms=(0.9,0.9))
    learn.save('done')

