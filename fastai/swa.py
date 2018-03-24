from .sgdr import *

class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        # greater than or gte?
        if self.epoch >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1
            
        self.epoch += 1
            
    def update_average_model(self):
        # update running average of parameters  
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        print(f'EPOCH {self.epoch}')
        for model_param, swa_param in zip(model_params, swa_params):
            print('BEFORE')
            print(model_param.data)
            print(swa_param.data)
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)
            print('AFTER')
            print(swa_param.data)
            
        
        