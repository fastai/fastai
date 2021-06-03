from fastai.basics import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *
from fastai.text.all import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
def pr(s):
    if rank_distrib()==0: print(s)

@call_parse
def main(
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=64,
    epochs:Param("Number of epochs", int)=5,
    fp16:  Param("Use mixed precision training", store_true)=False,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    "Training of IMDB classifier."
    path = rank0_first(untar_data, URLs.IMDB)
    dls = TextDataLoaders.from_folder(path, bs=bs, valid='test')

    for run in range(runs):
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        learn = rank0_first(text_classifier_learner, dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        # Workaround: In PyTorch 1.4, need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of text_classifier_learner.fine_tune()
        if num_distrib() > 1 and torch.__version__.startswith("1.4"): DistributedTrainer.fup = True
        with learn.distrib_ctx(): # distributed traing requires "-m fastai.launch"
            learn.fine_tune(epochs, lr)


