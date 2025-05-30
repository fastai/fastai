import pytest
from fastai.text.all import *

def test_custom_aws_lstm_has_layers():
    "Test if AWS_LSTM model is compatible with get_text_classifier"
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv')
    dls = TextDataLoaders.from_df(df, text_col='text', label_col='label')
    
    model = get_text_classifier(AWS_LSTM, vocab_sz=len(dls.vocab), n_class=dls.c)
    assert hasattr(model[0].module, 'layers'), "AWS_LSTM missing required `.layers` attribute"
