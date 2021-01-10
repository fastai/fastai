from fastai.basics import *
from fastai.tabular.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def pr(s):
    if rank_distrib()==0: print(s)

def get_dls(path):
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    return dls

@call_parse
def main(
    epochs:Param("Number of epochs", int)=5,
    fp16:  Param("Use mixed precision training", store_true)=False,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    "Training of Tabular data 'ADULT_SAMPLE'."
    path = rank0_first(untar_data,URLs.ADULT_SAMPLE)
    dls = get_dls(path)
    pr(f'epochs: {epochs};')

    for run in range(runs):
        pr(f'Run: {run}')
        learn = tabular_learner(dls, metrics=accuracy)
        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        n_gpu = torch.cuda.device_count()
        ctx = learn.distrib_ctx if num_distrib() and n_gpu else learn.parallel_ctx
        with ctx(): learn.fit_one_cycle(epochs)

