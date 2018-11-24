import pytest, torch, fastai
from fastai import *  
from fastai.callbacks import *
from fastai.vision import *
from fastai.text import *
from fastai.tabular import *
from fastai.collab import *

def test_model_summary_vision():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    model_summary(learn.model)
    
def test_model_summary_text():
    path = untar_data(URLs.IMDB_SAMPLE)
    data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
    learn = language_model_learner(data_lm, pretrained_model=None)
    model_summary(learn.model)
    
def test_model_summary_tabular():
    path = untar_data(URLs.ADULT_SAMPLE)
    dep_var = '>=50k'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    procs = [FillMissing, Categorify]
    df = pd.read_csv(path/'adult.csv')
    data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                               .split_by_idx(list(range(800,1000)))
                               .label_from_df(cols=dep_var)
                               .databunch())
    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
    model_summary(learn.model)
    
def test_model_summary_collab():
    path = untar_data(URLs.ML_SAMPLE)
    ratings = pd.read_csv(path/'ratings.csv')
    series2cat(ratings, 'userId', 'movieId')
    data = CollabDataBunch.from_df(ratings, seed=42)
    y_range = [0,5.5]
    learn = collab_learner(data, n_factors=50, y_range=y_range)
    model_summary(learn.model)
    
def test_model_summary_nn_module():
    model_summary(nn.Conv2d(16,32,3,padding=1))
    