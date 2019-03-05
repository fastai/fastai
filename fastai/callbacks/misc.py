" Miscellaneous callbacks "

from fastai.callback import Callback

class StopAfterNBatches(Callback):
    "Stop training after n batches of the first epoch."
    def __init__(self, n_batch:int=2):
        self.stop,self.n_batch = False,n_batch

    def on_batch_end(self, iteration, **kwargs):
        if iteration >= self.n_batch: return {'stop_epoch': True, 'stop_training': True}
