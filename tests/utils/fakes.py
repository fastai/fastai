from fastai.basics import *

class RandomItem(ItemBase):
    "An random `ItemBase` of size `noise_sz`."
    def __init__(self, *sizes): self.obj,self.data = sizes,torch.randn(*sizes)
    def __str__(self):  return f'Size {self.obj}\n{self.data}'

    @classmethod
    def from_val(cls, t):
        res = cls(*t.size())
        res.data = t
        return res

class RandomLabel(ItemBase):
    "An random `ItemBase` of size `noise_sz`."
    def __init__(self, y_range): self.obj,self.data = y_range,torch.randint(*y_range, ())
    def __str__(self):  return f'{self.data.item()}'

    @classmethod
    def from_val(cls, t):
        res = cls([0,t.item()+1])
        res.data = t
        return res

class RandomLabelList(ItemList):
    "An random `ItemBase` of size `noise_sz`."
    def __init__(self, items, y_range:Collection[int]=None, **kwargs):
        super().__init__(items, **kwargs)
        self.y_range = y_range
        self.copy_new.append('y_range')

    def get(self, i):  return RandomLabel(self.y_range)
    def reconstruct(self, t): return RandomLabel.from_val(t)

class RandomItemList(ItemList):
    _label_cls = RandomLabelList
    def __init__(self, items, sizes:Collection[int]=None, **kwargs):
        super().__init__(items, **kwargs)
        self.sizes = sizes
        self.copy_new.append('sizes')

    def get(self, i): return RandomItem(*self.sizes)
    def reconstruct(self, t): return RandomItem.from_val(t)

    def show_xys(self, xs, ys, **kwargs):
        res = [f'{x},{y}' for x,y in zip(xs, ys)]
        print('\n'.join(res))

    def show_xyzs(self, xs, ys, zs, **kwargs):
        res = [f'{x},{y},{z}' for x,y,z in zip(xs, ys, zs)]
        print('\n'.join(res))

def fake_basedata(n_in:int=5,batch_size:int=5, train_length:int=None, valid_length:int=None):
    if train_length is None: train_length = 2 * batch_size
    if valid_length is None: valid_length = batch_size

    return torch.empty([train_length+valid_length, n_in]).random_(-10, 10)


def fake_data(n_in:int=5, n_out:int=4, batch_size:int=5, train_length:int=None, valid_length:int=None) -> DataBunch:
    if train_length is None: train_length = 2 * batch_size
    if valid_length is None: valid_length = batch_size
    return (RandomItemList([0] * (train_length+valid_length), sizes=[n_in])
                .split_by_idx(list(range(valid_length)))
                .label_const(0., y_range=[0,n_out])
                .databunch(bs=batch_size))

def fake_learner(n_in:int=5, n_out:int=4, batch_size:int=5, train_length:int=None, valid_length:int=None, layer_group_count:int=1) -> Learner:
    data = fake_data(n_in=n_in, n_out=n_out, batch_size=batch_size, train_length=train_length, valid_length=valid_length)
    additional = [nn.Sequential(nn.Linear(n_in, n_in)) for _ in range(layer_group_count - 1)]
    final = [nn.Sequential(nn.Linear(n_in, n_out))]
    layer_groups = additional + final 
    model = nn.Sequential(*layer_groups)
    learner = Learner(data, model)
    learner.layer_groups = layer_groups
    return learner
