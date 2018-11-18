"`fastai.data` loads and manages datasets with `DataBunch`"
from .torch_core import *
from .layers import MSELossFlat

DatasetType = Enum('DatasetType', 'Train Valid Test Single')
__all__ = ['DataBunch', 'DeviceDataLoader', 'DatasetType']

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
            if not self.skip_size1 or y.size(0) != 1: yield self.proc_batch(b)

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
        def _create_dl(dl, **kwargs):
            return DeviceDataLoader(dl, self.device, self.tfms, collate_fn, **kwargs)
        self.train_dl = _create_dl(train_dl, skip_size1=True)
        self.valid_dl = _create_dl(valid_dl)
        self.single_dl = _create_dl(DataLoader(valid_dl.dataset, batch_size=1, num_workers=0))
        self.test_dl  = _create_dl(test_dl) if test_dl is not None else None
        self.path = Path(path)

    def __repr__(self)->str:
        return f'{self.__class__.__name__};\nTrain: {self.train_ds};\nValid: {self.valid_ds};\nTest: {self.test_ds}'

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Dataset=None, path:PathOrStr='.', bs:int=64,
               num_workers:int=defaults.cpus, tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
               collate_fn:Callable=data_collate)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `tfms` for `Dataset`, `tfms` for `DataLoader`."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        val_bs = (bs*3)//2
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,val_bs,val_bs), (True,False,False))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)

    def __getattr__(self,k:int)->Any: return getattr(self.train_dl, k)
    def dl(self, ds_type:DatasetType=DatasetType.Valid)->DeviceDataLoader:
        "Returns appropriate `Dataset` for validation, training, or test (`ds_type`)."
        return (self.train_dl if ds_type == DatasetType.Train else
                self.test_dl if ds_type == DatasetType.Test else
                self.valid_dl if ds_type == DatasetType.Valid else
                self.single_dl)

    @property
    def dls(self):
        res = [self.train_dl, self.valid_dl, self.single_dl]
        return res if not self.test_dl else res + [self.test_dl]

    def add_tfm(self,tfm:Callable)->None:
        for dl in self.dls: dl.add_tfm(tfm)

    def show_batch(self, rows:int=None, ds_type:DatasetType=DatasetType.Train, **kwargs)->None:
        "Show a batch of data in `ds_type` on a few `rows`."
        dl = self.dl(ds_type)
        b_idx = next(iter(dl.batch_sampler))
        if rows is None: rows = int(math.sqrt(len(b_idx)))
        ds = dl.dataset
        ds[0][0].show_batch(b_idx, rows, ds, **kwargs)

    @property
    def train_ds(self)->Dataset: return self.train_dl.dl.dataset
    @property
    def valid_ds(self)->Dataset: return self.valid_dl.dl.dataset
    @property
    def loss_func(self)->Dataset: return getattr(self.train_ds, 'loss_func', F.nll_loss)

    @property
    def test_ds(self)->Dataset:
        return self.test_dl.dl.dataset if self.test_dl is not None else None

