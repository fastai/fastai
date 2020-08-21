from fastai.basics import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastscript import *
from fastai.text.all import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=64,
    epochs:Param("Number of epochs", int)=5,
    fp16:  Param("Use mixed precision training", int)=0,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    "Training of IMDB classifier."

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if gpu is None: gpu = list(range(n_gpu))[0] 
        torch.cuda.set_device(gpu)
    else:
        n_gpu = None

    path = rank0_first(lambda:untar_data(URLs.IMDB))
    dls = TextDataLoaders.from_folder(path, bs=bs, valid='test')

    for run in range(runs):
        print(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        learn = rank0_first(lambda: text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy))

        if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        # TODO: DataParallel would hit floating point error, disabled for now.
        # if gpu is None and n_gpu: ctx = partial(learn.parallel_ctx, device_ids=list(range(n_gpu)))

        # Workaround: In PyTorch 1.4, need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of text_classifier_learner.fine_tune()

        if num_distrib() > 1 and torch.__version__.startswith("1.4"): DistributedTrainer.fup = True

        with learn.distrib_ctx(cuda_id=gpu): # distributed traing requires "-m fastai.launch"
            print(f"Training in distributed data parallel context on GPU {gpu}", flush=True)
            learn.fine_tune(epochs, lr)


