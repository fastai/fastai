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
   nvidia-smi dmon
   ```
