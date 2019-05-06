import os
import torch

RUN_SLOW = os.environ.get("RUN_SLOW", False)
RUN_CUDA = torch.cuda.is_available()