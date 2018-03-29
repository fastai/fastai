import collections
import itertools
from concurrent.futures import ThreadPoolExecutor
import time

class LazyThreadPoolExecutor(ThreadPoolExecutor):
    def map(self, fn, *iterables, timeout=None, chunksize=1, prefetch=None):
        """
        Collects iterables lazily, rather than immediately.
        Docstring same as parent: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor
        Implmentation taken from this PR: https://github.com/python/cpython/pull/707
        """
        if timeout is not None: end_time = timeout + time.time()
        if prefetch is None: prefetch = self._max_workers
        if prefetch < 0: raise ValueError("prefetch count may not be negative")
        argsiter = zip(*iterables)
        fs = collections.deque(self.submit(fn, *args) for args in itertools.islice(argsiter, self._max_workers+prefetch))
        # Yield must be hidden in closure so that the futures are submitted before the first iterator value is required.
        def result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    res = fs[0].result() if timeout is None else fs[0].result(end_time-time.time())
                    # Got a result, future needn't be cancelled
                    del fs[0]
                    # Dispatch next task before yielding to keep pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))
                    yield res
            finally:
                for future in fs: future.cancel()
        return result_iterator()