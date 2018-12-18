from .torch_core import *
from .basic_train import Learner,LearnerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

__all__ = ['DistributedRecorder', 'DistributedTrainer', 'read_metrics', 'setup_distrib']

def make_async(b:Tuple[Tensor,Tensor]):
    return [o.to(o.device, non_blocking=True) for o in b]

@dataclass
class DistributedTrainer(LearnerCallback):
    cuda_id:int
    _order = -20 #Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        self.learn.model = DistributedDataParallel(self.learn.model, device_ids=[self.cuda_id],
                                                   output_device=self.cuda_id)
        train_sampler = DistributedSampler(self.learn.data.train_ds)
        self.learn.data.train_dl = self.learn.data.train_dl.new(shuffle=False, sampler=train_sampler)
        self.learn.data.train_dl.add_tfm(make_async)
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl.add_tfm(make_async)
        self.rank = int(os.environ['RANK'])
        self.learn.recorder.silent = (self.rank != 0)

    def on_train_end(self, **kwargs):
        self.learn.data.train_dl.remove_tfm(make_async)
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl.remove_tfm(make_async)

@dataclass
class DistributedRecorder(LearnerCallback):
    cuda_id:int
    cache_dir:PathOrStr='tmp'

    def on_train_begin(self, **kwargs):
        os.makedirs(self.learn.path/self.cache_dir, exist_ok=True)

    def on_epoch_end(self, **kwargs): self.save_stats()
    def on_train_end(self, **kwargs): self.save_stats()

    def save_stats(self):
        cache_path,recorder = self.learn.path/self.cache_dir,self.learn.recorder
        np.save(cache_path/f'losses_{self.cuda_id}', np.array(recorder.losses))
        stats = np.array([[v] + m for v,m in zip(recorder.val_losses,recorder.metrics)])
        np.save(cache_path/f'metrics_{self.cuda_id}', stats)

def _learner_distributed(learn:Learner, cuda_id:int, cache_dir:PathOrStr='tmp'):
    learn.callbacks.append(DistributedTrainer(learn, cuda_id))
    learn.callbacks.append(DistributedRecorder(learn, cuda_id, cache_dir))
    return learn

Learner.distributed = _learner_distributed

def read_metrics(cache_path:PathOrStr, n_gpus:int, reduce:bool=True):
    losses,metrics = [],[]
    for i in range(n_gpus):
        losses.append(np.load(cache_path/f'losses_{i}.npy')[None])
        metrics.append(np.load(cache_path/f'metrics_{i}.npy')[None])
    if reduce:
        losses,metrics = np.concatenate(losses,0),np.concatenate(metrics,0)
        return losses.mean(0),metrics.mean(0)
    return losses,metrics

def setup_distrib(gpu:Any=None):
    if gpu is None: return gpu
    gpu = int(gpu)
    torch.cuda.set_device(int(gpu))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu

