import pytest
from fastai import *
from fastai.text import *

def test_order_preds():
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv')
    train_df, valid_df = df[:200], df[950:]
    data_clas = TextClasDataBunch.from_df(path, train_df, valid_df)
    learn = text_classifier_learner(data_clas)
    preds = learn.get_preds(ordered=True)
    true_value = np.array([data_clas.train_ds.c2i[o] for o in valid_df.iloc[:,0]])
    assert np.all(torch.Tensor.numpy(preds[1]) == true_value)

