---
title: Working with GPU
---

## GPU Monitoring

Here is how to poll the status of your GPU(s) in a variety of ways from your terminal:

1. watch the processes using GPU(s) and the current state of your GPU(s):

   ```
   watch -n 1 nvidia-smi
   ```

2. Watch the usage stats as their change:

   ```
   nvidia-smi --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -t
   ```

   This way is useful as you can see the trace of changes, rather than just the current state shown by `nvidia-smi` executed without any arguments.

   * To see what other options you can query run: `nvidia-smi --help-query-gpu`.

   * `-l 1` will update every 1 sec (`--loop. You can increase that number to do it less frequently.

   * `-f filename` will log into a file, but you won't be able to see the output. So it's better to use `nvidia-smi ... | tee filename` instead, which will show the output and log the results as well.

   * if you'd like the program to stop logging after running for 3600 seconds, run it as: `timeout -t 3600 nvidia-smi ...`

   For more details, please, see [Useful nvidia-smi Queries](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries).

3. Similar to the above, but shows the stats as a percentage:

   ```
   nvidia-smi dmon -s u
   ```
   which shows the essentials (usage and memory). If you would like all of the stats, run it without arguments:
   ```
   nvidia-smi dmon
   ```


## Accessing NVIDIA GPU Info Programmatically

While watching `nvidia-smi` running in your terminal is handy, sometimes you want to do more than that. And that's where API access comes in handy. The following tools provide that.


### pynvml

`nvidia-ml-py3` provides Python 3 bindings for nvml c-lib (NVIDIA Management Library), which allows you to query the library directly, without needing to go through `nvidia-smi`. Therefore this module is much faster than the wrappers around `nvidia-smi`.

The bindings are implemented with `Ctypes`, so this module is `noarch` - it's just pure python.

Installation:

* Pypi:
```
pip3 install nvidia-ml-py3
```
* Conda:
```
conda install nvidia-ml-py3 -c fastai
```

Examples:

Print the memory stats for the first GPU card:
```
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)
```

List the available GPU devices:

```
from pynvml import *
nvmlInit()
try:
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("Device", i, ":", nvmlDeviceGetName(handle))
except NVMLError as error:
    print(error)
```

And here is a usage example via a sample module `nvidia_smi`:

```
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
```



### GPUtil

GPUtil is a wrapper around `nvidia-smi`, and requires the latter to function before it can be used.

Installation: `pip3 install gputil`.

And here is a usage example:

```
import GPUtil as GPU
GPUs = GPU.getGPUs()
gpu = GPUs[0]
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
```

For more details see: https://github.com/anderskm/gputil

For more details see: https://github.com/nicolargo/nvidia-ml-py3


### gpustat

`nvidia-smi` like monitor, but a compact one. It relies on [pynvml](https://pythonhosted.org/nvidia-ml-py/) to talk to the nvml layer.

Installation: `pip3 install gpustat`.

And here is a usage example:

```
gpustat -cp -i --no-color
```

For more details see: https://github.com/wookayin/gpustat


## GPU Memory Notes


### Unusable GPU RAM per process

As soon as you start using `cuda`, your GPU loses about 0.5GB RAM per process. For example this code consumes 0.5GB GPU RAM:
```
import torch
torch.ones((1, 1)).cuda()
```
This GPU memory is not accessible to your program's needs and it's not re-usable between processes. If you run two processes, each executing code on `cuda`, each will consume 0.5GB GPU RAM from the get going.

This fixed chunk of memory is used by `cuDNN` kernels (~300MB) and `pytorch` (the rest) for its internal needs.


### Cached Memory

`pytorch` normally caches GPU RAM it previously used to re-use it at a later time. So the output from `nvidia-smi` could be incorrect in that you may have more GPU RAM available than it reports. You can reclaim this cache with:
```
import torch
torch.cuda.empty_cache()
```

If you have more than one process using the same GPU, the cached memory from one process is not accessible to the other. The above code executed by the first process will solve this issue and make the freed GPU RAM available to the other process.


### Reusing GPU RAM

How can we do a lot of experimentation in a given jupyter notebook w/o needing to restart the kernel all the time? You can delete the variables that hold the memory, can call `import gc; gc.collect()` to reclaim memory by deleted objects with circular references, optionally (if you have just one process) calling `torch.cuda.empty_cache()` and you can now re-use the GPU memory inside the same kernel.

To automate this process, and get various stats on memory consumption, you can use [IPyExperiments](https://github.com/stas00/ipyexperiments). Other than helping you to reclaim general and GPU RAM, it is also helpful with efficiently tuning up your notebook parameters to avoid `cuda: out of memory` errors and detecting various other memory leaks.
