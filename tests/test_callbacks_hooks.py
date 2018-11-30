import pytest, torch, fastai
from fastai import *
from fastai.callbacks import *
from fastai.vision import *
from fastai.text import *
from fastai.tabular import *
from fastai.collab import *

def test_model_summary_vision():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    model_summary(learn)

@pytest.mark.xfail(reason = "Expected Fail, text models not supported yet.")
def test_model_summary_text():
    path = untar_data(URLs.IMDB_SAMPLE)
    data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
    learn = language_model_learner(data_lm, pretrained_model=None)
    model_summary(learn)

def test_model_summary_tabular():
    path = untar_data(URLs.ADULT_SAMPLE)
    dep_var = '>=50k'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    procs = [FillMissing, Categorify]
    df = pd.read_csv(path/'adult.csv')
    data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs, bs=2)
                               .split_by_idx(list(range(800,1000)))
                               .label_from_df(cols=dep_var)
                               .databunch())
    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
    model_summary(learn)

def test_model_summary_collab():
    path = untar_data(URLs.ML_SAMPLE)
    ratings = pd.read_csv(path/'ratings.csv')
    series2cat(ratings, 'userId', 'movieId')
    data = CollabDataBunch.from_df(ratings, seed=42, bs=2)
    y_range = [0,5.5]
    learn = collab_learner(data, n_factors=50, y_range=y_range)
    model_summary(learn)

def test_model_summary_nn_module():
    model_summary(nn.Conv2d(16,16,3,padding=1))

def test_model_summary_nn_modules():
    class BasicBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = conv2d(16,16,3,1)
            self.conv2 = conv2d(16,16,3,1)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x
    model_summary(BasicBlock())

