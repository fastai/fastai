from .torch_core import *
from torch.optim import Optimizer

__all__ = ['StatScope', 'Statistic', 'AvgSquare', 'GeneralOptimizer']

StatScope = Enum('StatScope', 'Global Group Layer Channel Weight')

@dataclass
class Statistic():
    scope: StatScope
    name: str
    init: Union[float, Tensor]
    beta: float = 0.9
    
    def new_step(self):                raise NotImplementedError
    def accumulate(self, val):         raise NotImplementedError
    def update(self, state, val=None): raise NotImplementedError

@dataclass
class AvgSquare(Statistic):
    
    def new_step(self): self.val,self.count = 0.,0
    def accumulate(self, val): 
        self.count += val.numel()
        self.val += val.pow(2).mean()
        
    def update(self, state, val=None):
        if self.scope == StatScope.Weight: return state.mul_(self.beta).addcmul_(1-self.beta, val, val)
        if self.scope == StatScope.Layer:  return self.beta * state + (1 - self.beta) * val.pow(2).mean()
        if self.count != 0: return self.beta * state + (1 - self.beta) * (self.val / self.count) ** 2
        return self.beta
        
class GeneralOptimizer(Optimizer):
    
    def __init__(self, params, defaults, stats=None):
        super().__init__(params, defaults)
        self.global_stats,self.group_stats,self.layer_stats,self.channel_stats,self.weight_stats = self._split_stats(stats)
        self.init_stats()
    
    def step(self, closure=None):
        self.update_stats()
        for pg in self.param_groups:
            for p in pg['params']: self.make_step(p, pg)
    
    def make_step(self, p, group):
        d_p = p.grad.data
        p.data.add_(-group['lr'], p.grad)

    def _split_stats(self, stats):
        return ([stat for stat in listify(stats) if stat.scope==scope] for scope in StatScope)
    
    def _init_stats(self, stats, data=None):
        return {stat.name: stat.init if data is None else torch.zeros_like(data) + stat.init for stat in stats}
        
    def init_stats(self):
        self.state.update(self._init_stats(self.global_stats))
        for i,pg in enumerate(self.param_groups):
            self.state[f'group{i}'] = self._init_stats(self.group_stats)
            for p in pg['params']:
                self.state[p] = self._init_stats(self.layer_stats)
                self.state[p].update(self._init_stats(self.weight_stats, p.data))
    
    def update_stats(self):
        for stat in self.global_stats: stat.new_step()
        for i,pg in enumerate(self.param_groups):
            for stat in self.group_stats: stat.new_step()
            for p in pg['params']:
                for stat in self.global_stats + self.group_stats: stat.accumulate(p.grad)
                for stat in self.layer_stats + self.weight_stats:  
                    self.state[p][stat.name] = stat.update(self.state[p][stat.name], p.grad)
            for stat in self.group_stats: 
                self.state[f'group{i}'][stat.name] = stat.update(self.state[f'group{i}'][stat.name])
        for stat in self.global_stats: self.state[stat.name] = stat.update(self.state[stat.name])