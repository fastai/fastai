from fastai.basics import *
from fastai.tabular.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastscript import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dls(path):
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    return dls

@call_parse
def main(
    gpu:   Param("GPU to run on", int)=None,
    epochs:Param("Number of epochs", int)=5,
    fp16:  Param("Use mixed precision training", int)=0,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    "Training of Tabular data 'ADULT_SAMPLE'."

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)

    path = untar_data(URLs.ADULT_SAMPLE)
    dls = get_dls(path)

    if not gpu: print(f'epochs: {epochs};')

    for run in range(runs):
        print(f'Run: {run}')

        learn = tabular_learner(dls, metrics=accuracy)
        if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        n_gpu = torch.cuda.device_count()
        ctx = learn.parallel_ctx if gpu is None and n_gpu else learn.distrib_ctx

        with partial(ctx, gpu)(): # distributed traing requires "-m fastai.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            learn.fit_one_cycle(epochs)
