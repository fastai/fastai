from fastai.basics import *

def fake_data(n_in:int=5, n_out:int=4, batch_size:int=5, train_length:int=None, valid_length:int=None):
    if train_length is None: train_length = 2 * batch_size
    if valid_length is None: valid_length = batch_size
    train_ds = TensorDataset(torch.randn(train_length, n_in), torch.randn(train_length, n_out))
    valid_ds = TensorDataset(torch.randn(valid_length, n_in), torch.randn(valid_length, n_out))
    return DataBunch.create(train_ds, valid_ds, bs=batch_size)

def fake_learner(n_in=5, n_out=4, batch_size:int=5, train_length:int=None, valid_length:int=None):
    data = fake_data(n_in=n_in, n_out=n_out, batch_size=batch_size, train_length=train_length, valid_length=valid_length)
    model = nn.Linear(n_in,n_out)
    return Learner(data, model, loss_func=nn.MSELoss())