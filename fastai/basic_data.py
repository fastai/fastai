"`fastai.data` loads and manages datasets with `DataBunch`"
from .torch_core import *

DatasetType = Enum('DatasetType', 'Train Valid Test')
__all__ = ['SingleClassificationDataset', 'LabelXYDataset', 'DataBunch', 'DatasetBase', 'DeviceDataLoader', 'LabelDataset', 'DatasetType']

class DatasetBase(Dataset):
    "Base class for all fastai datasets."
    def __init__(self, c:int): self.c,self.item = c,None
    def __len__(self): return len(getattr(self, 'x', [1]))
    def set_item(self,item): self.item = item
    def clear_item(self): self.item = None
    def __repr__(self): return f'{type(self).__name__} of len {len(self)}'
    def new(self, *args, **kwargs):
        "Create a new dataset using `self` as a template"
        return self.__class__(*args, **kwargs)

    def _get_x(self,i): return self.x[i]
    def _get_y(self,i): return self.y[i]

    def __getitem__(self, i):
        if self.item is None: return self._get_x(i),self._get_y(i)
        else: return self.item,0

class LabelDataset(DatasetBase):
    "Base class for fastai datasets that do classification, mapped according to `classes`."
    def __init__(self, classes:Collection, class2idx:Dict[Any,int]=None):
        self.classes  = classes
        self.class2idx = class2idx
        if class2idx is None: self.class2idx = {v:k for k,v in enumerate(self.classes)}
        super().__init__(len(classes))

class LabelXYDataset(LabelDataset):
    "Minimal `LabelDataset` which returns whatever `x` and `y` you pass in"
    def __init__(self, x:Collection, y:Collection, classes:Optional[Collection[Any]]=None):
        super().__init__(classes=classes)
        self.x,self.y  = np.array(x),np.array(y)

class SingleClassificationDataset(DatasetBase):
    "A `Dataset` that contains no data, only `classes`, mainly used for inference with `set_item`"
    def __init__(self, classes:Collection[str]):
        self.classes = classes
        super().__init__(len(classes))

def DataLoader___getattr__(dl, k:str)->Any: return getattr(dl.dataset, k)
DataLoader.__getattr__ = DataLoader___getattr__

@dataclass
class DeviceDataLoader():
    "Bind a `DataLoader` to a `torch.device`."
    dl: DataLoader
    device: torch.device
    tfms: List[Callable]=None
    collate_fn: Callable=data_collate
    skip_size1:bool=False
    def __post_init__(self):
        self.dl.collate_fn=self.collate_fn
        self.tfms = listify(self.tfms)

    def __len__(self)->int: return len(self.dl)
    def __getattr__(self,k:str)->Any: return getattr(self.dl, k)

    @property
    def batch_size(self):   return self.dl.batch_size
    @batch_size.setter
    def batch_size(self,v): self.dl.batch_size = v

    @property
    def num_workers(self):   return self.dl.num_workers
    @num_workers.setter
    def num_workers(self,v): self.dl.num_workers = v

    def add_tfm(self,tfm:Callable)->None:    self.tfms.append(tfm)
    def remove_tfm(self,tfm:Callable)->None: self.tfms.remove(tfm)

    def proc_batch(self,b:Tensor)->Tensor:
        "Proces batch `b` of `TensorImage`."
        b = to_device(b, self.device)
        for f in listify(self.tfms): b = f(b)
        return b

    def __iter__(self):
        "Process and returns items from `DataLoader`."
        for b in self.dl: 
            y = b[1][0] if is_listy(b[1]) else b[1]
            if not self.skip_size1 or y.size(0) != 1:
                yield self.proc_batch(b)

    def one_batch(self)->Collection[Tensor]:
        "Get one batch from the data loader."
        w = self.num_workers
        self.num_workers = 0
        it = iter(self)
        try:     return next(it)
        finally: self.num_workers = w

    @classmethod
    def create(cls, dataset:Dataset, bs:int=64, shuffle:bool=False, device:torch.device=defaults.device,
               tfms:Collection[Callable]=tfms, num_workers:int=defaults.cpus, collate_fn:Callable=data_collate, **kwargs:Any):
        "Create DeviceDataLoader from `dataset` with `batch_size` and `shuffle`: processs using `num_workers`."
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)

class DataBunch():
    "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
    def __init__(self, train_dl:DataLoader, valid_dl:DataLoader, test_dl:Optional[DataLoader]=None,
                 device:torch.device=None, tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.',
                 collate_fn:Callable=data_collate):
        "Bind `train_dl`,`valid_dl` and`test_dl` to `device`. tfms are DL tfms (normalize). `path` is for models."
        self.tfms = listify(tfms)
        self.device = defaults.device if device is None else device
        assert not isinstance(train_dl,DeviceDataLoader)
        self.train_dl = DeviceDataLoader(train_dl, self.device, self.tfms, collate_fn, skip_size1=True)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, self.tfms, collate_fn)
        self.test_dl  = DeviceDataLoader(test_dl, self.device, self.tfms, collate_fn) if test_dl is not None else None
        self.path = Path(path)

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Dataset=None, path:PathOrStr='.', bs:int=64,
               num_workers:int=defaults.cpus, tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
               collate_fn:Callable=data_collate)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `tfms` for `Dataset`, `tfms` for `DataLoader`."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,bs*2,bs*2), (True,False,False))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)

    def __getattr__(self,k:int)->Any: return getattr(self.train_dl, k)
    def dl(self, ds_type:DatasetType=DatasetType.Valid)->DeviceDataLoader:
        "Returns appropriate `Dataset` for validation, training, or test (`ds_type`)."
        return (self.train_dl if ds_type == DatasetType.Train else
                self.test_dl if ds_type == DatasetType.Test else
                self.valid_dl)

    def add_tfm(self,tfm:Callable)->None:
        self.train_dl.add_tfm(tfm)
        self.valid_dl.add_tfm(tfm)
        if self.test_dl: self.test_dl.add_tfm(tfm)

    @property
    def train_ds(self)->Dataset: return self.train_dl.dl.dataset
    @property
    def valid_ds(self)->Dataset: return self.valid_dl.dl.dataset
    @property
    def loss_func(self)->Dataset: return getattr(self.train_ds, 'loss_func', F.nll_loss)

    @property
    def test_ds(self)->Dataset:
        assert self.test_dl is not None, "You didn't specify a test set for this DataBunch."
        return self.test_dl.dl.dataset
