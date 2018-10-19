"`fastai.data` loads and manages datasets with `DataBunch`"
from .torch_core import *

__all__ = ['DataBunch', 'DatasetBase', 'DeviceDataLoader', 'LabelDataset']

class DatasetBase(Dataset):
    "Base class for all fastai datasets."
    def __len__(self): return len(self.x)
    @property
    def c(self):
        "Number of classes expressed by dataset y variable."
        return self.y.shape[-1] if len(self.y.shape)>1 else 1
    def __repr__(self): return f'{type(self).__name__} of len {len(self)}'

class LabelDataset(DatasetBase):
    "Base class for fastai datasets that do classification."
    @property
    def c(self):
        "Number of classes expressed by dataset y variable."
        return len(self.classes)

@dataclass
class DeviceDataLoader():
    "Bind a `DataLoader` to a `torch.device`."
    dl: DataLoader
    device: torch.device
    tfms: List[Callable]=None
    collate_fn: Callable=data_collate
    def __post_init__(self):
        self.dl.collate_fn=self.collate_fn
        self.tfms = listify(self.tfms)

    def __len__(self)->int: return len(self.dl)
    def __getattr__(self,k:str)->Any: return getattr(self.dl, k)

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
        for b in self.dl: yield self.proc_batch(b)

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
        self.train_dl = DeviceDataLoader(train_dl, self.device, self.tfms, collate_fn)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, self.tfms, collate_fn)
        self.test_dl  = DeviceDataLoader(test_dl,  self.device, self.tfms, collate_fn) if test_dl else None
        self.path = Path(path)

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Dataset=None, path:PathOrStr='.', bs:int=64,
               num_workers:int=defaults.cpus, tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
               collate_fn:Callable=data_collate)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,bs*2,bs*2), (True,False,False))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)

    def __getattr__(self,k:int)->Any: return getattr(self.train_ds, k)
    def holdout(self, is_test:bool=False)->DeviceDataLoader:
        "Returns correct holdout `Dataset` for test vs validation (`is_test`)."
        return self.test_dl if is_test else self.valid_dl

    def add_tfm(self,tfm:Callable)->None:
        self.train_dl.add_tfm(tfm)
        self.valid_dl.add_tfm(tfm)
        if self.test_dl: self.test_dl.add_tfm(tfm)

    @property
    def train_ds(self)->Dataset: return self.train_dl.dl.dataset
    @property
    def valid_ds(self)->Dataset: return self.valid_dl.dl.dataset
    @property
    def loss_func(self)->Dataset: return self.train_ds.loss_func