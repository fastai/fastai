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



### Python 3 Bindings for the NVIDIA Management Library

`nvidia-ml-py3` provides Python 3 bindings for nvml c-lib (NVIDIA Management Library), which allows you to query the library directly, without needing to go through `nvidia-smi`.

Installation: `pip3 install nvidia-ml-py3`.

And here is a usage example:

```
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
```

For more details see: https://github.com/nicolargo/nvidia-ml-py3


### gpustat

`nvidia-smi` like monitor, but a compact one. It relies on [pynvml](https://pythonhosted.org/nvidia-ml-py/) to talk to the nvml layer.

Installation: `pip3 install gpustat`.

And here is a usage example:

```
gpustat -cp -i --no-color
```

For more details see: https://github.com/wookayin/gpustat
