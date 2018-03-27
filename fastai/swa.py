import torch
from .sgdr import *
from .core import *


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1
            
        self.epoch += 1
            
    def update_average_model(self):
        # update running average of parameters  
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)
            

def _is_bn(module): return isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    
def _is_bn_flag(module, flag): 
    if _is_bn(module): flag[0] = True
    
def uses_bn(model):
    flag = [False]
    model.apply(lambda module: _is_bn_flag(module, flag))
    return flag[0]

def _reset_bn(module):
    if _is_bn(module):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def reset_bn_stats(model):
    model.apply(_reset_bn)
    
def _get_momenta(module, momenta):
    if _is_bn(module):
        momenta[module] = module.momentum
    
def _set_momenta(module, momenta):
    if _is_bn(module):
        module.momentum = momenta[module]
        
def fix_batchnorm(swa_model, train_dl):
    if not uses_bn(swa_model):
        return

    swa_model.train()

    reset_bn_stats(swa_model)

    momenta = {}
    swa_model.apply(lambda module: _get_momenta(module, momenta))            

    inputs_seen = 0

    for (*x,y) in iter(train_dl):
        xs = V(x)
        batch_size = xs[0].size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in momenta.keys():
            module.momentum = momentum
                
        swa_model.apply(lambda module: _set_momenta(module, momenta))

        swa_model(*xs)        
        
        inputs_seen += batch_size
                
    swa_model.apply(lambda module: _set_momenta(module, momenta))            