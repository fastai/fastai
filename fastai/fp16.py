import torch
import torch.nn as nn


class FP16(nn.Module):
    def __init__(self, module): 
        super(FP16, self).__init__()
        self.module = batchnorm_to_fp32(module.half())
        
    def forward(self, input): 
        return self.module(input.half())
    
    def load_state_dict(self, *inputs, **kwargs):
        self.module.load_state_dict(*inputs, **kwargs)

    def state_dict(self, *inputs, **kwargs):
        return self.module.state_dict(*inputs, **kwargs)

def batchnorm_to_fp32(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_fp32(child)
    return module

def copy_model_to_fp32(m, optim):
    """  Creates a fp32 copy of model parameters and sets optimizer parameters
    """
    fp32_params = [m_param.clone().type(torch.cuda.FloatTensor).detach() for m_param in m.parameters()]
    optim_groups = [group['params'] for group in optim.param_groups]
    iter_fp32_params = iter(fp32_params)
    for group_params in optim_groups:
        for i in range(len(group_params)):
            fp32_param = next(iter_fp32_params)
            fp32_param.requires_grad = group_params[i].requires_grad
            group_params[i] = fp32_param
    return fp32_params

def copy_fp32_to_model(m, fp32_params):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        m_param.data.copy_(fp32_param.data)

def update_fp32_grads(fp32_params, m):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        if fp32_param.grad is None:
            fp32_param.grad = nn.Parameter(fp32_param.data.new().resize_(*fp32_param.data.size()))
        fp32_param.grad.data.copy_(m_param.grad.data)
