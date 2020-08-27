import pandas as pd
import torch
from torch import as_tensor,Tensor,ByteTensor,LongTensor,FloatTensor,HalfTensor,DoubleTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SequentialSampler,RandomSampler,Sampler,BatchSampler
from torch.utils.data import IterableDataset,get_worker_info
from torch.utils.data._utils.collate import default_collate,default_convert

# Python 3.8 changed to 'spawn' but that doesn't work with PyTorch DataLoader w n_workers>0
import platform
if platform.system()=='Darwin': multiprocessing.set_start_method('fork', force=True)

