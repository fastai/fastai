from .torch_core import *
from torch.optim import Optimizer
import types

__all__ = ['StatScope', 'Statistic', 'AvgStatistic', 'AvgSquare', 'GeneralOptimizer']

#TODO: Channel
StatScope = Enum('StatScope', 'Global Group Layer Channel Weight')

@dataclass
class Statistic():
    name:str
    param:float=0.9  # e.g. for exp moving average
    scope:StatScope=StatScope.Weight
    init:float=0.  # starting value

    @property
    def buf(self): return f'{self.name}_buffer'

    def new_step(self):
        "Set state when computing statistics for Global or Group"
        raise NotImplementedError

    def accumulate(self, val):
        "Add `val` to statistic"
        raise NotImplementedError

    def update(self, state, param, val=None):
        "Update state with accumlated, or `val` (if `Weight` or `Layer` scope)"
        raise NotImplementedError

@dataclass
class AvgStatistic(Statistic):
    decay:bool=False
    debias:bool=False
    def new_step(self): self.val,self.count = 0.,0

    def accumulate(self, val):
        self.count += 1
        self.val += self._get_val(val)

    def _get_val1(self, val): return val.mean()
    def _get_val2(self, state, val, param): return state.add_(1-param, val) if self.decay else state.add_(val)
    def _get_val3(self, state, val, param):
        v = val.view(val.size(0), -1).mean(1)
        return state.add_(1-param, v) if self.decay else state.add(v)

    def update(self, state, param, val=None):
        if self.scope == StatScope.Weight:
            # `state` is a tensor
            return self._get_val2(state.mul_(param), val, param)
        if self.scope == StatScope.Channel:
            # `state` is a tensor of size n_channels
            return self._get_val3(state.mul_(param), val, param)
        # For everything else, `state` is a scalar
        if self.scope == StatScope.Layer:
            return state.lerp_(self._get_val1(val), 1-param)
        if self.count != 0:
            return state.lerp_(self.val/self.count, 1-param)
        return state

class AvgSquare(AvgStatistic):

    def __init__(self, name:str, param:float=0.9, scope=StatScope.Weight, init:float=0., decay:bool=True, debias:bool=False):
        super().__init__(name, param=param, scope=scope, init=init, decay=decay, debias=debias)

    def _get_val1(self, val): return torch.norm(val).pow(2)/val.numel()
    def _get_val2(self, state, val, param): return state.addcmul_(1-param, val, val) if self.decay else state.addcmul_(val, val)
    def _get_val3(self, state, val, param):
        v = val.view(val.size(0), -1).mean(1)
        return state.addcmul_(1-param, v, v) if self.decay else state.addcmul_(v, v)

class GeneralOptimizer(Optimizer):
    def __init__(self, params, stats=None, on_step:Callable=None):
        defaults = {s.name:s.param for s in listify(stats) if s.name is not None}
        super().__init__(params, defaults)
        self.global_stats,self.group_stats,self.layer_stats,self.channel_stats,self.weight_stats = self._split_stats(stats)
        self.init_stats()
        if on_step is not None: self.on_step = types.MethodType(on_step, self)

    def step(self, closure=None):
        self.update_stats()
        for i,pg in enumerate(self.param_groups):
            for p in pg['params']:
                if p.grad is not None: self.on_step(p, pg, i)

    def on_step(self, p, group, group_idx): p.data.add_(-group['lr'], p.grad.data)

    def _split_stats(self, stats):
        return ([stat for stat in listify(stats) if stat.scope==scope] for scope in StatScope)

    def _init_stats(self, stats, data=None):
        return {stat.buf: tensor(stat.init) if data is None
                else torch.zeros_like(data) + stat.init for stat in stats}

    def init_stats(self):
        self.state['global'] = self._init_stats(self.global_stats)
        for i,pg in enumerate(self.param_groups):
            self.state[f'group{i}'] = self._init_stats(self.group_stats)
            for p in pg['params']:
                self.state[p] = self._init_stats(self.layer_stats)
                self.state[p].update(self._init_stats(self.channel_stats, p.data.view(p.data.size(0), -1).mean(1)))
                self.state[p].update(self._init_stats(self.weight_stats, p.data))

    def _set_bufs(self, p, stats, pg, val=None):
        d = self.state[p]
        for stat in stats: d[stat.buf] = stat.update(d[stat.buf], pg[stat.name], val=val)

    def update_stats(self):
        for stat in self.global_stats: stat.new_step()
        for i,pg in enumerate(self.param_groups):
            for stat in self.group_stats: stat.new_step()
            for p in pg['params']:
                if p.grad is not None:
                    for stat in self.global_stats + self.group_stats: stat.accumulate(p.grad.data)
                    self._set_bufs(p, self.layer_stats+self.channel_stats+self.weight_stats, pg, p.grad.data)
            self._set_bufs(f'group{i}', self.group_stats, pg)
        self._set_bufs('global', self.global_stats, self.param_groups[0])

