---
title: Working with GPU
---

## GPU Monitoring

Here is how to poll the status of your GPU(s) in a variety of ways from your terminal:

* Watch the processes using GPU(s) and the current state of your GPU(s):

   ```
   watch -n 1 nvidia-smi
   ```

* Watch the usage stats as their change:

   ```
   nvidia-smi --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
   ```

   This way is useful as you can see the trace of changes, rather than just the current state shown by `nvidia-smi` executed without any arguments.

   * To see what other options you can query run: `nvidia-smi --help-query-gpu`.

   * `-l 1` will update every 1 sec (`--loop. You can increase that number to do it less frequently.

   * `-f filename` will log into a file, but you won't be able to see the output. So it's better to use `nvidia-smi ... | tee filename` instead, which will show the output and log the results as well.

   * if you'd like the program to stop logging after running for 3600 seconds, run it as: `timeout -t 3600 nvidia-smi ...`

   For more details, please, see [Useful nvidia-smi Queries](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries).

   Most likely you will just want to track the memory usage, so this is probably sufficient:

   ```
   nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1
   ```

* Similar to the above, but show the stats as percentages:

   ```
   nvidia-smi dmon -s u
   ```
   which shows the essentials (usage and memory). If you would like all of the stats, run it without arguments:
   ```
   nvidia-smi dmon
   ```
   To find out the other options, use:
   ```
   nvidia-smi dmon -h
   ```

* [nvtop](https://github.com/Syllo/nvtop)

   Nvtop stands for NVidia TOP, a (h)top like task monitor for NVIDIA GPUs. It can handle multiple GPUs and print information about them in a htop familiar way.

   It shows the processes, and also visually displays the memory and gpu stats.

   This application requires building it from source (needing `gcc`, `make`, et al), but the instructions are easy to follow and it is quick to build.


* [gpustat](https://github.com/wookayin/gpustat)

   `nvidia-smi` like monitor, but a compact one. It relies on [pynvml](https://pythonhosted.org/nvidia-ml-py/) to talk to the nvml layer.

   Installation: `pip3 install gpustat`.

   And here is a usage example:

   ```
   gpustat -cp -i --no-color
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

This library is now a `fastai` dependency, so you can use it directly.

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


### py3nvml

This is another fork of `nvidia-ml-py3`, supplementing it with [extra useful utils](https://github.com/fbcotter/py3nvml).

note: there is no `py3nvml` conda package in its main channel, but it is available on pypi.


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

https://github.com/FrancescAlted/ipython_memwatcher



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


### GPU RAM Fragmentation

If you encounter an error similar to the following:

```
RuntimeError: CUDA out of memory.
Tried to allocate 350.00 MiB
(GPU 0; 7.93 GiB total capacity; 5.73 GiB already allocated;
324.56 MiB free; 1.34 GiB cached)
```

You may ask yourself, if there is 0.32 GB free and 1.34 GB cached (i.e. 1.66 GB total of unused memory), how can it not allocate 350 MB? This happens because of memory fragmentation.

For the sake of this example let's assume that you have a function that allocates as many GBs of GPU RAM as its argument specifies:

```
def allocate_gb(n_gbs): ...
```

And you have an 8GB GPU card and no process is using it, so when a process is starting it's the first one to use it.

If you do the following sequence of GPU RAM allocations:

```
                    # total used | free | 8gb of RAM
                    #        0GB | 8GB  | [________]
x1 = allocate_gb(2) #        2GB | 6GB  | [XX______]
x2 = allocate_gb(4) #        6GB | 2GB  | [XXXXXX__]
del x1              #        4GB | 4GB  | [__XXXX__]
x3 = allocate_gb(3) # failure to allocate 3GB w/ RuntimeError: CUDA out of memory
```

despite having a total of 4GB of free GPU RAM (cached and free), the last command will fail, because it can't get 3GB of contiguous memory.

You can conclude from this example, that it's crucial to always free up anything that's on CUDA as soon as you're done using it, and only then move new objects to CUDA. Normally a simple `del obj` does the trick. However, if your object has circular references in it, it will not be freed despite the `del()` call, until `gc.collect()` will not be called by python. And until the latter happens, it'll still hold the allocated GPU RAM! And that also means that in some situations you may want to call `gc.collect()` yourself.

If you want to educate yourself on how and when the python garbage collector gets automatically invoked see [gc](https://docs.python.org/3/library/gc.html#gc.get_threshold) and [this](https://rushter.com/blog/python-garbage-collector/).




### pytorch Tensor Memory Tracking

Show all the currently allocated Tensors:
```
import torch
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except: pass
```
Note, that gc will not contain some tensors that consume memory [inside autograd](https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/22).

Here is a good [discussion on this topic](https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741) with more related code snippets.
