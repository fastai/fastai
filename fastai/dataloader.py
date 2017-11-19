import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from .imports import *
import collections,sys,traceback,threading

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)

def np_collate(batch):
    b = batch[0]
    if isinstance(b, (np.ndarray, np.generic)): return np.stack(batch)
    elif isinstance(b, (int, float)): return np.array(batch)
    elif isinstance(b, string_classes): return batch
    elif isinstance(b, collections.Mapping):
        return {key: np_collate([d[key] for d in batch]) for key in b}
    elif isinstance(b, collections.Sequence):
        return [np_collate(samples) for samples in zip(*batch)]
    raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))


def get_tensor(batch, pin):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = torch.from_numpy(batch).contiguous()
        return batch.pin_memory() if pin else batch
    elif isinstance(batch, string_classes): return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin) for sample in batch]
    raise TypeError("batch must contain numbers, dicts or lists; found {}"
                     .format(type(batch)))


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=None, collate_fn=np_collate, pin_memory=False, drop_last=False):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.collate_fn,self.pin_memory,self.drop_last = collate_fn,pin_memory,drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self): return len(self.batch_sampler)

    def get_batch(self, indices): return self.collate_fn([self.dataset[i] for i in indices])

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as e:
            for batch in e.map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory)

