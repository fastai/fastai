##To train a language model on Wikitext-103
##`python train_wt103.py fwd` for the forward pretrained model in fastai
##`python train_wt103.py bwd --backwards True` for the backward pretrained model in fastai

from fastai.text import *
from fastai.script import *
from fastprogress import fastprogress

@call_parse
def main(
        name:Param("Name of the experiment", str, opt=False),
        gpu:Param("GPU to run on", int)=0,
        lr: Param("Learning rate", float)=1e-2,
        drop_mult: Param("Dropouts multiplicator", float)=0.1,
        wd: Param("Weight Decay", float)=0.1,
        epochs: Param("Number of epochs", int)=12,
        bs: Param("Batch size", int)=256,
        bptt: Param("Bptt", int)=80,
        backwards: Param("Backward model", bool)=False
        ):
    "Training on Wikitext 103"
    path = Config().data_path()/'wikitext-103'
    fastprogress.SAVE_PATH = f'{name}.txt' #Save the output of the progress bar in {name}.txt
    torch.cuda.set_device(gpu)
    data = load_data(path, bs=bs, bptt=bptt, backwards=backwards)
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, pretrained=False,
                                   metrics=[accuracy, Perplexity()])
    learn = learn.to_fp16(clip=0.1)

    learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7), div_factor=10, wd=wd)

    learn = learn.to_fp32()
    learn.save(f'{name}', with_opt=False)
    learn.data.vocab.save(path/f'{name}_vocab.pkl')
