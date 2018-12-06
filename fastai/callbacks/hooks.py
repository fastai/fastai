"Hooks provide extensibility at the model level."
from ..torch_core import *
from ..callback import *
from ..basic_train import *
from ..basic_data import *

__all__ = ['ActivationStats', 'Hook', 'HookCallback', 'Hooks', 'hook_output', 'hook_outputs',
           'model_sizes', 'num_features_model', 'model_summary', 'dummy_eval', 'dummy_batch']

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms:Collection[nn.Module], hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self,i:int)->Hook: return self.hooks[i]
    def __len__(self)->int: return len(self.hooks)
    def __iter__(self): return iter(self.hooks)
    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()

def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else o if is_listy(o) else list(o)

def hook_output (module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)

def hook_outputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)

class HookCallback(LearnerCallback):
    "Callback that can be used to register hooks on `modules`. Implement the corresponding function in `self.hook`."
    def __init__(self, learn:Learner, modules:Sequence[nn.Module]=None, do_remove:bool=True):
        super().__init__(learn)
        self.modules,self.do_remove = modules,do_remove

    def on_train_begin(self, **kwargs):
        "Register the `Hooks` on `self.modules`."
        if not self.modules:
            self.modules = [m for m in flatten_model(self.learn.model)
                            if hasattr(m, 'weight')]
        self.hooks = Hooks(self.modules, self.hook)

    def on_train_end(self, **kwargs):
        "Remove the `Hooks`."
        if self.do_remove: self.remove()

    def remove(self): self.hooks.remove()
    def __del__(self): self.remove()

class ActivationStats(HookCallback):
    "Callback that record the mean and std of activations."

    def on_train_begin(self, **kwargs):
        "Initialize stats."
        super().on_train_begin(**kwargs)
        self.stats = []

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        return o.mean().item(),o.std().item()
    def on_batch_end(self, train, **kwargs):
        "Take the stored results and puts it in `self.stats`"
        if train: self.stats.append(self.hooks.stored)
    def on_train_end(self, **kwargs): 
        "Polish the final result."
        self.stats = tensor(self.stats).permute(2,1,0)

def dummy_batch(m: nn.Module, size:tuple=(64,64))->Tensor:
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).zero_().requires_grad_(False)

def dummy_eval(m:nn.Module, size:tuple=(64,64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))

def model_sizes(m:nn.Module, size:tuple=(64,64))->Tuple[Sizes,Tensor,Hooks]:
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]

def num_features_model(m:nn.Module)->int:
    "Return the number of output features for `model`."
    return model_sizes(m)[-1][1]

def total_params(m:nn.Module)->int:
    params, trainable = 0, False
    if hasattr(m, "weight") and hasattr(m.weight, "size"):
         params += m.weight.numel()
         trainable = m.weight.requires_grad
    if hasattr(m, "bias") and hasattr(m.bias, "size"): params += m.bias.numel()
    return params, trainable

def hook_params(modules:Collection[nn.Module])->Hooks:
    return Hooks(modules, lambda m, i, o: total_params(m))

def params_size(m: nn.Module, size: tuple = (64, 64))->Tuple[Sizes, Tensor, Hooks]:
    "Pass a dummy input through the model to get the various sizes. Returns (res,x,hooks) if `full`"
    if isinstance(m, Learner):
        x = m.data.one_batch(detach=False, denorm=False)[0]
        m = m.model
    elif isinstance(m, nn.Module):
        ch_in = in_channels(m)
        x = next(m.parameters()).new(1, ch_in, *size)
    else: raise TypeError('You should either pass in a Learner or nn.Module')
    hooks_outputs = hook_outputs(flatten_model(m))
    hooks_params = hook_params(flatten_model(m))
    hooks = zip(hooks_outputs, hooks_params)
    x = m.eval()(*x) if is_listy(x) else m.eval()(x)
    output_size = [(o.stored.shape) for o in hooks_outputs]
    params = [o.stored for o in hooks_params]
    params, trainables = map(list,zip(*params))
    return (output_size, params, trainables, hooks)

def get_layer_name(layer:nn.Module)->str:
    return str(layer.__class__).split(".")[-1].split("'")[0]

def layers_info(m:Collection[nn.Module]) -> Collection[namedtuple]:
    func = lambda m:list(map(get_layer_name, flatten_model(m)))
    layers_names = func(m.model) if isinstance(m, Learner) else func(m)
    layers_sizes, layers_params, layers_trainable, _ = params_size(m)
    layer_info = namedtuple('Layer_Information', ['Layer', 'OutputSize', 'Params', 'Trainable'])
    return list(map(layer_info, layers_names, layers_sizes, layers_params, layers_trainable))

def model_summary(m:Collection[nn.Module], n:int=70):
    "Print a summary of `m` using a output text width of `n` chars"
    info = layers_info(m)
    header = ["Layer (type)", "Output Shape", "Param #", "Trainable"]
    print("=" * n)
    print(f"{header[0]:<20} {header[1]:<20} {header[2]:<10} {header[3]:<10}")
    print("=" * n)
    total_params = 0
    total_trainable_params = 0
    for layer, size, params, trainable in info:
        total_params += int(params)
        total_trainable_params += int(params) * trainable
        params, size, trainable = str(params), str(list(size)), str(trainable)
        print(f"{layer:<20} {size:<20} {params:<10} {trainable:<10}")
        print("_" * n)
    print("\nTotal params: ", total_params)
    print("Total trainable params: ", total_trainable_params)
    print("Total non-trainable params: ", total_params - total_trainable_params)

Learner.summary = model_summary

