import pytest
from fastai import *
from fastai.text import *

def test_order_preds():
    path = untar_data(URLs.IMDB_SAMPLE)
    data_lm = TextLMDataBunch.from_csv(path)
    data_clas = TextClasDataBunch.from_csv(path, vocab=data_lm.train_ds.vocab)
    learn = RNNLearner.classifier(data_clas)
    preds = learn.get_preds()
    true_value = pd.read_csv(path/'valid.csv', header=None).iloc[:,0]
    assert np.all(torch.Tensor.numpy(preds[1]) == true_value)
    
