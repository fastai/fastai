" Utils for modifying what is displayed in notbooks and command line"
import fastai
import fastprogress

class progress_diabled():
    ''' Context manager to disable the progress update bar and Recorder print'''
    def __init__(self,learner:Learner):
        self.learn = learner
        self.orig_callback_fns = copy(learner.callback_fns)
        def __enter__(self):
            #silence progress bar
            fastprogress.fastprogress.NO_BAR = True
            fastai.basic_train.master_bar, fastai.basic_train.progress_bar = fastprogress.force_console_behavior()
            self.learn.callback_fns[0] = partial(Recorder,add_time=True,silent=True) #silence recorder
            return self.learn

        def __exit__(self,type,value,traceback):
            fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar,progress_bar
            self.learn.callback_fns = self.orig_callback_fns