"Provides convenient callbacks for Learners that write model images, metrics/losses, stats and histograms to Tensorboard"
from ..basic_train import Learner
from ..basic_data import DatasetType, DataBunch
from ..vision import Image
from ..callbacks import LearnerCallback
from ..core import *
from ..torch_core import *
from threading import Thread, Event
from time import sleep
from queue import Queue
import statistics
import torchvision.utils as vutils
from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter


__all__=['LearnerTensorboardWriter', 'GANTensorboardWriter', 'ImageGenTensorboardWriter']


#---Example usage (applies to any of the callbacks)--- 
# proj_id = 'Colorize'
# tboard_path = Path('data/tensorboard/' + proj_id)
# learn.callback_fns.append(partial(GANTensorboardWriter, base_dir=tboard_path, name='GanLearner'))

class LearnerTensorboardWriter(LearnerCallback):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500, stats_iters:int=100):
        super().__init__(learn=learn)
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.loss_iters = loss_iters
        self.hist_iters = hist_iters
        self.stats_iters = stats_iters
        self.hist_writer = HistogramTBWriter()
        self.stats_writer = ModelStatsTBWriter()
        self.data = None
        self.metrics_root = '/metrics/'
        self._update_batches_if_needed()

    def _update_batches_if_needed(self):
        # one_batch function is extremely slow with large datasets.  This is an optimization.
        # Note that also we want to always show the same batches so we can see changes 
        # in tensorboard
        update_batches = self.data is not self.learn.data

        if update_batches:
            self.data = self.learn.data
            self.trn_batch = self.learn.data.one_batch(
                ds_type=DatasetType.Train, detach=True, denorm=False, cpu=False)
            self.val_batch = self.learn.data.one_batch(
                ds_type=DatasetType.Valid, detach=True, denorm=False, cpu=False)

    def _write_model_stats(self, iteration:int):
        self.stats_writer.write(
            model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    def _write_training_loss(self, iteration:int, last_loss:Tensor):
        scalar_value = to_np(last_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_weight_histograms(self, iteration:int):
        self.hist_writer.write(
            model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    #TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?
    def _write_metrics(self, iteration:int, last_metrics:MetricsList, start_idx:int=2):
        recorder = self.learn.recorder

        for i, name in enumerate(recorder.names[start_idx:]):
            if len(last_metrics) < i+1: return
            scalar_value = last_metrics[i]
            tag = self.metrics_root + name
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def on_batch_end(self, last_loss:Tensor, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        if iteration % self.loss_iters == 0:
            self._write_training_loss(iteration=iteration, last_loss=last_loss)

        if iteration % self.hist_iters == 0:
            self._write_weight_histograms(iteration=iteration)

    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop
    def on_backward_end(self, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        if iteration % self.stats_iters == 0:
            self._write_model_stats(iteration=iteration)

    def on_epoch_end(self, last_metrics:MetricsList, iteration:int, **kwargs):
        self._write_metrics(iteration=iteration, last_metrics=last_metrics)

# TODO:  We're overriding almost everything here.  Seems like a good idea to question that ("is a" vs "has a")
class GANTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500,
                 stats_iters:int=100, visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters,
                         hist_iters=hist_iters, stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageTBWriter()
        self.gen_stats_updated = True
        self.crit_stats_updated = True

    # override
    def _write_weight_histograms(self, iteration:int):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic
        self.hist_writer.write(
            model=generator, iteration=iteration, tbwriter=self.tbwriter, name='generator')
        self.hist_writer.write(
            model=critic, iteration=iteration, tbwriter=self.tbwriter, name='critic')

    # override
    def _write_model_stats(self, iteration:int):
        trainer = self.learn.gan_trainer
        generator = trainer.generator
        critic = trainer.critic

        # Don't want to write stats when model is not iterated on and hence has zeroed out gradients
        gen_mode = trainer.gen_mode

        if gen_mode and not self.gen_stats_updated:
            self.stats_writer.write(
                model=generator, iteration=iteration, tbwriter=self.tbwriter, name='gen_model_stats')
            self.gen_stats_updated = True

        if not gen_mode and not self.crit_stats_updated:
            self.stats_writer.write(
                model=critic, iteration=iteration, tbwriter=self.tbwriter, name='crit_model_stats')
            self.crit_stats_updated = True

    # override
    def _write_training_loss(self, iteration:int, last_loss:Tensor):
        trainer = self.learn.gan_trainer
        recorder = trainer.recorder

        if len(recorder.losses) > 0:
            scalar_value = to_np((recorder.losses[-1:])[0])
            tag = self.metrics_root + 'train_loss'
            self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write(self, iteration:int):
        trainer = self.learn.gan_trainer
        #TODO:  Switching gen_mode temporarily seems a bit hacky here.  Certainly not a good side-effect.  Is there a better way?
        gen_mode = trainer.gen_mode

        try:
            trainer.switch(gen_mode=True)
            self.img_gen_vis.write(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch,
                                                    iteration=iteration, tbwriter=self.tbwriter)
        finally:                                      
            trainer.switch(gen_mode=gen_mode)

    # override
    def on_batch_end(self, iteration:int, **kwargs):
        super().on_batch_end(iteration=iteration, **kwargs)
        if iteration == 0: return
        if iteration % self.visual_iters == 0:
            self._write(iteration=iteration)

    # override
    def on_backward_end(self, iteration:int, **kwargs):
        if iteration == 0: return
        self._update_batches_if_needed()

        #TODO:  This could perhaps be implemented as queues of requests instead but that seemed like overkill. 
        # But I'm not the biggest fan of maintaining these boolean flags either... Review pls.
        if iteration % self.stats_iters == 0:
            self.gen_stats_updated = False
            self.crit_stats_updated = False

        if not (self.gen_stats_updated and self.crit_stats_updated):
            self._write_model_stats(iteration=iteration)


class ImageGenTensorboardWriter(LearnerTensorboardWriter):
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500,
                 stats_iters: int = 100, visual_iters: int = 100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, hist_iters=hist_iters,
                         stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageTBWriter()

    def _write(self, iteration:int):
        self.img_gen_vis.write(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch,
                                                  iteration=iteration, tbwriter=self.tbwriter)

    # override
    def on_batch_end(self, iteration:int, **kwargs):
        super().on_batch_end(iteration=iteration, **kwargs)
        if iteration == 0: return

        if iteration % self.visual_iters == 0:
            self._write(iteration=iteration)


#------PRIVATE-----------

class TBWriteRequest(ABC):
    def __init__(self, tbwriter: SummaryWriter, iteration:int):
        super().__init__()
        self.tbwriter = tbwriter
        self.iteration = iteration

    @abstractmethod
    def write(self):
        pass   


# SummaryWriter writes tend to block quite a bit.  This gets around that and greatly boosts performance.
# Not all tensorboard writes are using this- just the ones that take a long time.  Note that the 
# SummaryWriter does actually use a threadsafe consumer/producer design ultimately to write to Tensorboard, 
# so writes done outside of this async loop should be fine.
class AsyncTBWriter():
    def __init__(self):
        super().__init__()
        self.stop_request = Event()
        self.queue = Queue()
        self.thread = Thread(target=self._queue_processor, daemon=True)
        self.thread.start()

    def request_write(self, request: TBWriteRequest):
        if self.stop_request.isSet():
            raise Exception('Close was already called!  Cannot perform this operation.')
        self.queue.put(request)

    def _queue_processor(self):
        while not self.stop_request.isSet():
            while not self.queue.empty():
                request = self.queue.get()
                request.write()
            sleep(0.2)

    #Provided this to stop thread explicitly or by context management (with statement) but thread should end on its own 
    # upon program exit, due to being a daemon.  So using this is probably unecessary.
    def close(self):
        self.stop_request.set()
        self.thread.join()

    def __enter__(self):
        # Nothing to do, thread already started.  Could start thread here to enforce use of context manager 
        # (but that sounds like a pain and a bit unweildy and unecessary for actual usage)
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

asyncTBWriter = AsyncTBWriter() 

class ModelImageSet():
    @staticmethod
    def get_list_from_model(learn:Learner, ds_type:DatasetType, batch:Tuple)->[]:
        image_sets = []
        x,y = batch[0],batch[1]
        preds = learn.pred_batch(ds_type=ds_type, batch=(x,y), reconstruct=True)
        
        for orig_px, real_px, gen in zip(x,y,preds):
            orig = Image(px=orig_px)
            real = Image(px=real_px)
            image_set = ModelImageSet(orig=orig, real=real, gen=gen)
            image_sets.append(image_set)

        return image_sets  

    def __init__(self, orig:Image, real:Image, gen:Image):
        self.orig = orig
        self.real = real
        self.gen = gen


class HistogramTBRequest(TBWriteRequest):
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.params = [(name, values.clone().detach().cpu()) for (name, values) in model.named_parameters()]
        self.name = name

    # override
    def write(self):
        try:
            for param_name, values in self.params:
                tag = self.name + '/weights/' + param_name
                self.tbwriter.add_histogram(tag=tag, values=values, global_step=self.iteration)
        except Exception as e:
            print(("Failed to write model histograms to Tensorboard:  {0}").format(e))

#If this isn't done async then this is sloooooow
class HistogramTBWriter():
    def __init__(self):
        super().__init__()

    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model'):
        request = HistogramTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)

class ModelStatsTBRequest(TBWriteRequest):
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.gradients = [x.grad.clone().detach().cpu() for x in model.parameters() if x.grad is not None]
        self.name = name
        self.gradients_root = '/gradients/'

    # override
    def write(self):
        try:
            if len(self.gradients) == 0: return

            gradient_nps = [to_np(x.data) for x in self.gradients]
            avg_norm = sum(x.data.norm() for x in self.gradients)/len(self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'avg_norm', scalar_value=avg_norm, global_step=self.iteration)

            median_norm = statistics.median(x.data.norm() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'median_norm', scalar_value=median_norm, global_step=self.iteration)

            max_norm = max(x.data.norm() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'max_norm', scalar_value=max_norm, global_step=self.iteration)

            min_norm = min(x.data.norm() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'min_norm', scalar_value=min_norm, global_step=self.iteration)

            num_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'num_zeros', scalar_value=num_zeros, global_step=self.iteration)

            avg_gradient = sum(x.data.mean() for x in self.gradients)/len(self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'avg_gradient', scalar_value=avg_gradient, global_step=self.iteration)

            median_gradient = statistics.median(x.data.median() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'median_gradient', scalar_value=median_gradient, global_step=self.iteration)

            max_gradient = max(x.data.max() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'max_gradient', scalar_value=max_gradient, global_step=self.iteration)

            min_gradient = min(x.data.min() for x in self.gradients)
            self.tbwriter.add_scalar(
                tag=self.name + self.gradients_root + 'min_gradient', scalar_value=min_gradient, global_step=self.iteration)
        except Exception as e:
            print(("Failed to write model stats to Tensorboard:  {0}").format(e))


class ModelStatsTBWriter():
    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model_stats'):
        request = ModelStatsTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)


class ImageTBRequest(TBWriteRequest):
    def __init__(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.image_sets = ModelImageSet.get_list_from_model(learn=learn, batch=batch, ds_type=ds_type)
        self.ds_type = ds_type

    # override
    def write(self):
        try:
            orig_images = []
            gen_images = []
            real_images = []

            for image_set in self.image_sets:
                orig_images.append(image_set.orig.px)
                gen_images.append(image_set.gen.px)
                real_images.append(image_set.real.px)

            prefix = self.ds_type.name

            self.tbwriter.add_image(
                tag=prefix + ' orig images', img_tensor=vutils.make_grid(orig_images, normalize=True), 
                global_step=self.iteration)
            self.tbwriter.add_image(
                tag=prefix + ' gen images', img_tensor=vutils.make_grid(gen_images, normalize=True), 
                global_step=self.iteration)
            self.tbwriter.add_image(
                tag=prefix + ' real images', img_tensor=vutils.make_grid(real_images, normalize=True), 
                global_step=self.iteration)
        except Exception as e:
            print(("Failed to write images to Tensorboard:  {0}").format(e))

#If this isn't done async then this is noticeably slower
class ImageTBWriter():
    def __init__(self):
        super().__init__()

    def write(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, iteration:int, tbwriter:SummaryWriter):
        self._write_for_dstype(learn=learn, batch=val_batch, iteration=iteration,
                             tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._write_for_dstype(learn=learn, batch=trn_batch, iteration=iteration,
                             tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _write_for_dstype(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        request = ImageTBRequest(learn=learn, batch=batch, iteration=iteration, tbwriter=tbwriter, ds_type=ds_type)
        asyncTBWriter.request_write(request)



