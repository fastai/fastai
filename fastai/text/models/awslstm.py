# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.text.all import *
from fastai.text.models.core import SequentialRNN, _model_meta
from fastai.text.models.awdlstm import awd_lstm_clas_split

# Define AWS_LSTM model
class AWS_LSTM(Module):
    "Custom LSTM encoder compatible with FastAI `get_text_classifier`"
    
    def __init__(self,
        vocab_sz:int,
        emb_sz:int=400,
        n_hid:int=1152,
        n_layers:int=3,
        pad_token:int=1,
        dropout:float=0.3,
        bidir:bool=False,
        **kwargs  # Accepts FastAI extra args
    ):
        _ = kwargs  # Avoids unused arg warning
        store_attr('vocab_sz, emb_sz, n_hid, n_layers, pad_token, dropout, bidir')
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.rnns = nn.ModuleList([
            nn.LSTM(emb_sz if i == 0 else n_hid, n_hid, num_layers=1,
                    bidirectional=bidir, batch_first=True)
            for i in range(n_layers)
        ])
        self.decoder = nn.Linear(n_hid * (2 if bidir else 1), n_hid)

        # Required for FastAI’s model splitter
        self.layers = [self.encoder] + list(self.rnns) + [self.decoder]

    def forward(self, x):
        x = self.encoder(x)
        for rnn in self.rnns:
            x, _ = rnn(x)
            x = F.dropout(x, self.dropout, self.training)
        x = x.mean(dim=1)
        return self.decoder(x)

# Define config for AWS_LSTM
aws_lstm_clas_config = dict(
    emb_sz=400,
    n_hid=1152,
    n_layers=3,
    pad_token=1,
    bidir=False,
    dropout=0.3,
    output_p=0.4,
    init=None,
    out_bias=True
)

# Register model in FastAI meta
_model_meta[AWS_LSTM] = {
    'config_clas': aws_lstm_clas_config,
    'split_clas': awd_lstm_clas_split,
    'emb_sz': 400,
    'hid_name': 'n_hid'  # Must be here
}


path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')
dls = TextDataLoaders.from_df(df, text_col='text', label_col='label')

model = get_text_classifier(AWS_LSTM, vocab_sz=len(dls.vocab), n_class=dls.c)

# Check your custom layers
print("Custom AWS_LSTM layers:", model[0].module.layers)

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
