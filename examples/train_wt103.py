##To train a language model on Wikitext-103
##`python train_wt103.py fwd` for the forward pretrained model in fastai
##`python train_wt103.py bwd --backwards True` for the backward pretrained model in fastai
## Takes 6 hours on a Titan RTX (24Gb RAM), adjust batch size and lr if less GPU RAM

from fastai.text import *
from fastai.script import *
from fastprogress import fastprogress

#Functions to parse WT103 in separate articles
def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0

def read_file(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', UNK)
    articles.append(current_article)
    return np.array(articles)

def create_data(path):
    train = read_file(path/'train.txt')
    valid = read_file(path/'valid.txt')
    test =  read_file(path/'test.txt')
    all_texts = np.concatenate([valid, train, test])
    df = pd.DataFrame({'texts':all_texts})
    del train ; del valid ; del test #Free RQM before tokenizing
    data = (TextList.from_df(df, path, cols='texts')
                    .split_by_idx(range(0,60))
                    .label_for_lm()
                    .databunch())
    data.save()

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
    if not (path/'data_save.pkl').is_file(): create_data(path)
    data = load_data(path, bs=bs, bptt=bptt, backwards=backwards)
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, pretrained=False,
                                   metrics=[accuracy, Perplexity()])
    learn = learn.to_fp16(clip=0.1)

    learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7), div_factor=10, wd=wd)

    learn = learn.to_fp32()
    learn.save(f'{name}', with_opt=False)
    learn.data.vocab.save(path/f'{name}_vocab.pkl')
