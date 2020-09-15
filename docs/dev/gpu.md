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

As soon as you start using CUDA, your GPU loses some 300-500MB RAM per process. The exact size seems to be depending on the card and CUDA version. For example, on GeForce GTX 1070 Ti (8GB), the following code, running on CUDA 10.0, consumes 0.5GB GPU RAM:
```
import torch
torch.ones((1, 1)).cuda()
```
This GPU memory is not accessible to your program's needs and it's not re-usable between processes. If you run two processes, each executing code on `cuda`, each will consume 0.5GB GPU RAM from the get going.

This fixed chunk of memory is used by [CUDA context](https://stackoverflow.com/questions/43244645/what-is-a-cuda-context).

### Cached Memory

`pytorch` normally caches GPU RAM it previously used to re-use it at a later time. So the output from `nvidia-smi` could be incorrect in that you may have more GPU RAM available than it reports. You can reclaim this cache with:
```
import torch
torch.cuda.empty_cache()
```

If you have more than one process using the same GPU, the cached memory from one process is not accessible to the other. The above code executed by the first process will solve this issue and make the freed GPU RAM available to the other process.

It also might be helpful to note that `torch.cuda.memory_cached()` doesn't show how much memory pytorch has free in the cache, but it just indicates how much memory it currently has allocated, with some of it being used and may be some being free. To measure how much free memory available to use is in the cache do: `torch.cuda.memory_cached()-torch.cuda.memory_allocated()`.



### Reusing GPU RAM

How can we do a lot of experimentation in a given jupyter notebook w/o needing to restart the kernel all the time? You can delete the variables that hold the memory, can call `import gc; gc.collect()` to reclaim memory by deleted objects with circular references, optionally (if you have just one process) calling `torch.cuda.empty_cache()` and you can now re-use the GPU memory inside the same kernel.

To automate this process, and get various stats on memory consumption, you can use [IPyExperiments](https://github.com/stas00/ipyexperiments). Other than helping you to reclaim general and GPU RAM, it is also helpful with efficiently tuning up your notebook parameters to avoid `CUDA: out of memory` errors and detecting various other memory leaks.

And also make sure you read the tutorial on `learn.purge` and its friends [here](/tutorial.resources.html), which provide an even better solution.


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

Except, this example isn't quite valid, because under the hood CUDA relocates physical pages, and makes them appear as if they are of a contiguous type of memory to pytorch. So in the example above it'll reuse most or all of those fragments as long as there is nothing else occupying those memory pages.

So for this example to be applicable to the CUDA memory fragmentation situation it needs to allocate fractions of a memory page, which currently for most CUDA cards is of 2MB. So if less than 2MB is allocated in the same scenario as this example, fragmentation will occur.

Given that GPU RAM is a scarce resource, it helps to always try free up anything that's on CUDA as soon as you're done using it, and only then move new objects to CUDA. Normally a simple `del obj` does the trick. However, if your object has circular references in it, it will not be freed despite the `del()` call, until `gc.collect()` will not be called by python. And until the latter happens, it'll still hold the allocated GPU RAM! And that also means that in some situations you may want to call `gc.collect()` yourself.

If you want to educate yourself on how and when the python garbage collector gets automatically invoked see [gc](https://docs.python.org/3/library/gc.html#gc.get_threshold) and [this](https://rushter.com/blog/python-garbage-collector/).


### Peak Memory Usage

If you were to run a GPU memory profiler on a function like `Learner` `fit()` you would notice that on the very first epoch it will cause a very large GPU RAM usage spike and then stabilize at a much lower memory usage pattern. This happens because the pytorch memory allocator tries to build the computational graph and gradients for the loaded model in the most efficient way. Luckily, you don't need to worry about this spike, since the allocator is smart enough to recognize when the memory is tight and it will be able to do the same with much less memory, just not as efficiently. Typically, continuing with the `fit()` example, the allocator needs to have at least as much memory as the 2nd and subsequent epochs require for the normal run.  You can read an excellent thread on this topic [here](https://discuss.pytorch.org/t/high-gpu-memory-usage-problem/34694).


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


### GPU Reset

If for some reason after exiting the python process the GPU doesn't free the memory, you can try to reset it (change 0 to the desired GPU ID):

```
sudo nvidia-smi --gpu-reset -i 0
```

When using multiprocessing, sometimes some of the client processes get stuck and go zombie and won't release the GPU memory. They also may become invisible to `nvidia-smi`, so that it reports no memory used, but the card is unusable and fails with OOM even when trying to create a tiny tensor on that card. In such a case locate the relevant processes with `fuser -v /dev/nvidia*`and kill them with `kill -9`.

This blog [post](https://jianchao-li.github.io/post/killing-pytorch-multi-gpu-training-the-safe-way/) suggests the following trick to arrange for the processes to cleanly exit on demand:
```
if os.path.isfile('kill.me'):
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
    exit(0)
```
After you add this code to the training iteration, once you want to stop it, just cd into the directory of the training program and run
```
touch kill.me
```

## Multi-GPU

### Order of GPUs

When having multiple GPUs you may discover that `pytorch` and `nvidia-smi` don't order them in the same way, so what `nvidia-smi` reports as `gpu0`, could be assigned to `gpu1` by `pytorch`. `pytorch` uses CUDA GPU ordering, which is done by [computing power](https://developer.nvidia.com/cuda-gpus) (higher computer power GPUs first).

If you want `pytorch` to use the PCI bus device order, to match `nvidia-smi`, set:

```
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

before starting your program (or put in your `~/.bashrc`).

If you just want to run on a specific gpu ID, you can use the `CUDA_VISIBLE_DEVICES` environment variable. It can be set to a single GPU ID or a list:

```
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2,3
```

If you don't set the environment variables in shell, you can set those in your code at the beginning of your program, with help of: `import os; os.environ['CUDA_VISIBLE_DEVICES']='2'`.

A less flexible way is to hardcode the device ID in your code, e.g. to set it to `gpu1`:
```
torch.cuda.set_device(1)
```
