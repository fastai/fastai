from .torch_core import *
from torch.optim import Optimizer

__all__ = ['StatScope', 'Statistic', 'AvgStatistic', 'AvgSquare', 'GeneralOptimizer']

#TODO: Channel
StatScope = Enum('StatScope', 'Global Group Layer Channel Weight')

@dataclass
class Statistic():
    name: str
    param: float = 0.9  # e.g. for exp moving average
    scope: StatScope = StatScope.Weight
    init: float = 0.0  # starting value

    @property
    def buf(self): return f'{self.name}_buffer'

    def new_step(self):
        "Set state when computing statistics for Global or Group"
        raise NotImplementedError

    def accumulate(self, val):
        "Add `val` to statistic"
        raise NotImplementedError

    def update(self, state, val=None):
        "Update state with accumlated, or `val` (if `Weight` or `Layer` scope)"
        raise NotImplementedError

class AvgStatistic(Statistic):
    def new_step(self): self.val,self.count = 0.,0

    def accumulate(self, val):
        self.count += 1
        self.val += self._get_val(val)

    def _get_val1(self, val): return val.mean()
    def _get_val2(self, state, val): return state.add_(val)

    def update(self, state, val=None):
        if self.scope == StatScope.Weight:
            # `state` is a tensor
            return self._get_val2(state.mul_(self.param), val)
        # For everything else, `state` is a scalar
        if self.scope == StatScope.Layer:
            return state.lerp_(self._get_val1(val), 1-self.param)
        if self.count != 0:
            return state.lerp_(self.val/self.count, 1-self.param)
        return state

class AvgSquare(AvgStatistic):
    def _get_val1(self, val): return torch.norm(val).pow(2)/val.numel()
    def _get_val2(self, state, val): return state.addcmul_(1-self.param, val, val)


class GeneralOptimizer(Optimizer):
    def __init__(self, params, stats=None):
        defaults = {s.name:s.param for s in listify(stats)}
        super().__init__(params, defaults)
        self.global_stats,self.group_stats,self.layer_stats,self.channel_stats,self.weight_stats = self._split_stats(stats)
        self.init_stats()

    def step(self, closure=None):
        self.update_stats()
        for i,pg in enumerate(self.param_groups):
            for p in pg['params']:
                if p.grad is not None: self.make_step(p, pg, i)

    def make_step(self, p, group, group_idx): p.data.add_(-group['lr'], p.grad.data)

    def _split_stats(self, stats):
        return ([stat for stat in listify(stats) if stat.scope==scope] for scope in StatScope)

    def _init_stats(self, stats, data=None):
        return {stat.buf: stat.init if data is None
                else torch.zeros_like(data) + stat.init for stat in stats}

    def init_stats(self):
        self.state['global'] = self._init_stats(self.global_stats)
        for i,pg in enumerate(self.param_groups):
            self.state[f'group{i}'] = self._init_stats(self.group_stats)
            for p in pg['params']:
                self.state[p] = self._init_stats(self.layer_stats)
                self.state[p].update(self._init_stats(self.weight_stats, p.data))

    def _set_bufs(self, p, stats, val=None):
        d = self.state[p]
        for stat in stats: d[stat.buf] = stat.update(d[stat.buf], val=val)

    def update_stats(self):
        for stat in self.global_stats: stat.new_step()
        for i,pg in enumerate(self.param_groups):
            for stat in self.group_stats: stat.new_step()
            for p in pg['params']:
                if p.grad is not None:
                    for stat in self.global_stats + self.group_stats: stat.accumulate(p.grad.data)
                    self._set_bufs(p, self.layer_stats+self.weight_stats, p.grad.data)
            self._set_bufs(f'group{i}', self.group_stats)
        self._set_bufs('global', self.global_stats)

