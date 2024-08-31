from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from .basics import *
from .callback.progress import ProgressCallback
from .data.load import _FakeLoader, _loaders
from .optimizer import OptimWrapper
try:
    from accelerate import Accelerator
except ModuleNotFoundError:
    Accelerator = None

__all__ = ['ParallelTrainer', 'setup_distrib', 'teardown_distrib', 'DistributedDL', 'DistributedTrainer', 'rank0_first']

def patch(func):
    "A simple decorator for patching methods."
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def store_attr(func):
    "A decorator to store attributes from function parameters."
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self
    return wrapper

def delegates(target, but=()):
    "A decorator to delegate arguments to another function."
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@patch
def reset(self: DataParallel):
    "Patch required `reset` call into `DataParallel`"
    if hasattr(self.module, 'reset'):
        self.module.reset()

class ParallelTrainer(Callback):
    "Wrap a model `DataParallel` automatically"
    run_after, run_before = TrainEvalCallback, Recorder

    def __init__(self, device_ids):
        self.device_ids = device_ids

    def before_fit(self):
        self.learn.model = DataParallel(self.learn.model, device_ids=self.device_ids)

    def after_fit(self):
        self.learn.model = self.learn.model.module

@patch
def to_parallel(self: Learner, device_ids=None):
    "Add `ParallelTrainer` callback to a `Learner`"
    self.add_cb(ParallelTrainer(device_ids))
    return self

@patch
def detach_parallel(self: Learner):
    "Remove `ParallelTrainer` callback from a Learner"
    self.remove_cb(ParallelTrainer)
    return self

@patch
@contextmanager
def parallel_ctx(self: Learner, device_ids=None):
    "A context manager to adapt a learner to train in data parallel mode."
    try:
        self.to_parallel(device_ids)
        yield self
    finally:
        self.detach_parallel()

@patch
def reset(self: DistributedDataParallel):
    "Patch required `reset` call into `DistributedDataParallel`"
    if hasattr(self.module, 'reset'):
        self.module.reset()

def setup_distrib(gpu=None):
    "Setup this process to participate in distributed training"
    if gpu is None:
        return gpu
    gpu = int(gpu)
    torch.cuda.set_device(gpu)
    if num_distrib() > 0:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu

def teardown_distrib():
    "Free distributed training resources"
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def _round_to_multiple(number, multiple):
    return int(math.ceil(number / multiple) * multiple)

class DistributedDL(TfmdDL):
    "A `TfmdDL` which splits a batch into equal size pieces for each worker"
    def __init__(self, dl, rank=None, world_size=None, device=None):
        if rank is None:
            rank = rank_distrib()
        if world_size is None:
            world_size = num_distrib()
        store_attr()
        if isinstance(dl, torch.utils.data.DataLoader):
            shuffle = isinstance(dl.sampler, torch.utils.data.RandomSampler)
            self.dl = DataLoader(dataset=dl.dataset, batch_size=dl.batch_size, num_workers=dl.num_workers,
                                pin_memory=dl.pin_memory, timeout=dl.timeout, shuffle=shuffle,
                                drop_last=dl.drop_last, persistent_workers=dl.persistent_workers)
        self.bs, self.drop_last, self.dataset, fake, self.num_workers, self.offs, self.pin_memory = \
            attrgetter('batch_size', 'drop_last', 'dataset', 'fake_l', 'num_workers', 'offs', 'pin_memory')(self.dl)
        if device is None:
            self.device = self.dl.device
        self.fake_l = _FakeLoader(self, fake.pin_memory, fake.num_workers, fake.timeout,
                                  persistent_workers=fake.persistent_workers,
                                  pin_memory_device=fake.pin_memory_device)

    def _broadcast(self, t, rank):
        "Broadcasts t from rank `rank` to all other ranks. Returns t so t is same for all ranks after call."
        t = LongTensor(t).cuda()  # nccl only works with cuda tensors
        torch.distributed.broadcast(t, rank)
        return t.cpu().tolist()

    def _to_detach(self, b, cpu=True, gather=True):
        return to_detach(b, cpu, gather)  # member func so we can override for test

    def __len__(self):
        return _round_to_multiple(len(self.dl), self.world_size) // self.world_size

    def get_idxs(self):
        idxs = list(self.dl.get_idxs())  # compute get_idxs in all ranks (we'll only use rank 0 but size must be consistent)
        idxs = self._broadcast(idxs, 0)  # broadcast and receive it from rank 0 to all
        self.n = len(idxs)  # we assumed n was dl.n but we really care about number of idxs
        # add extra samples to make it evenly divisible
        self.n_padded = _round_to_multiple(self.n, self.world_size)
        idxs += (idxs * (self.n_padded // self.n))[:self.n_padded - self.n]  # idx needs to be repeated when n_padded >> n
        # slice padded idxs so that each rank gets self.n_padded // self.world_size tensors
        return idxs[self.rank * self.n_padded // self.world_size:(self.rank + 1) * self.n_padded // self.world_size]

    def before_iter(self):
        self.i = 0
        self.dl.before_iter()

    def randomize(self):
        self.dl.randomize()

    def after_batch(self, b):
        self.i += find_bs(b)
        return self.dl.after_batch(b)

    def after_iter(self):
        self.dl.after_iter()

    def create_batches(self, samps):
        return self.dl.create_batches(samps)

    def to_detach(self, b, cpu=True, gather=True):
        b = self._to_detach(b, cpu, gather)
        def _inner(b):
            if b.ndim > 0:
                # for each rank, compute overflow of read idxs vs self.n and accumulate them to unpad totals after gathering
                n = sum([min(0, max(-len(b) // self.world_size,
                                    self.n - (self.i + r * self.n_padded // self.world_size))) for r in range(self.world_size)])
                b = b[:n or None]
            return b
        return apply(_inner, b) if gather and all(hasattr(self, o) for o in ('i', 'n', 'n_padded')) else b

class DistributedTrainer(Callback):
    "Wrap `model` in `DistributedDataParallel` and `dls` in `DistributedDL`"
    order = 11

    @delegates(Accelerator, but=["mixed_precision", "fp16", "log_with", "logging_dir", "step_scheduler_with_optimizer"])
    def __init__(self,
                 sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                 **kwargs):
        store_attr(self, **kwargs)
        self.accelerator = Accelerator(**kwargs)

    def before_fit(self):
        self.learn.model = self.accelerator.prepare(
            nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model
        )
        self.old_dls = list(self.dls)
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
        if rank_distrib():
            self.learn.logger = noop

    def _wrap_dl(self, dl):
        return dl if isinstance(dl, DistributedDL) else DistributedDL(dl, device=self.learn.model.device)

    def _backward(self):
        self.accelerator.backward(self.learn.loss_grad)

    def before_train(self):
        self.learn.dl = self._wrap_dl(self.learn.dl)

    def before_validate(self):
        self.learn.dl = self._wrap_dl(self.learn.dl)

    def after_fit(self):
        self.learn.model = self.learn.model.module

    def teardown(self):
        self.accelerator.end_training()

def rank0_first(f):
    "A decorator to run `f` only on rank 0."
    @wraps(f)
    def wrapper(*args, **kwargs):
        if rank_distrib() == 0:
            return f(*args, **kwargs)
    return wrapper
