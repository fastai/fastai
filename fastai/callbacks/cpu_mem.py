" Memory profiling callbacks "

import tracemalloc, threading, torch, time
from ..utils.mem import *
from ..basic_train import *
from ..torch_core import *

class CpuPeakMemMetric(LearnerCallback):
    "Callback that measures used and peaked general and CPU memory."

    _order = -20  # Needs to run before the recorder

    def peak_monitor_start(self):
        self.peak_monitoring = True

        # start RAM tracing
        tracemalloc.start()

        # this thread samples RAM usage as long as the current epoch of the fit loop is running
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def peak_monitor_stop(self):
        tracemalloc.stop()
        self.peak_monitoring = False

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1
        while True:
            if not self.peak_monitoring: break
            time.sleep(0.001)  # 1msec

    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['cpu used', 'cpu_peak'])

    def on_epoch_begin(self, **kwargs): self.peak_monitor_start()

    def on_epoch_end(self, last_metrics, **kwargs):
        cpu_used, cpu_peak = list(map(lambda x: float(x / 2 ** 20), tracemalloc.get_traced_memory()))
        self.peak_monitor_stop()
        # The numbers are deltas in MBs (beginning of the epoch and the end)
        return add_metrics(last_metrics, [cpu_used, cpu_peak])
