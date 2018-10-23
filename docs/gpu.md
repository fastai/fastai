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
   nvidia-smi --query-gpu=timestamp,pstate,temperature.gpu,memory.total,memory.free,memory.used --format=csv -l 5
   ```

   This way is useful as you can see the trace of changes, rather than just the current state shown by `nvidia-smi` executed without any arguments.

   To see what other options you can query run: `nvidia-smi --help-query-gpu`.

3. Similar to the above, but shows the stats as a percentage

   ```
   nvidia-smi dmon
   ```
