"Implements [mixup](https://arxiv.org/abs/1710.09412) training method"
from ..torch_core import *
from ..callback import *
from ..basic_train import Learner

@dataclass
class MixUpCallback(Callback):
    "Callback that creates the mixed-up input and target."
    learner:Learner
    alpha:float=0.4
    stack_x:bool=False
    stack_y:bool=True
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1)
            new_target = last_target * lambd + y1 * (1-lambd)
        return (new_input, new_target)  

class MixUpLoss(nn.Module):
    "Adapt the loss function to go with mixup."
    
    def __init__(self, crit):
        super().__init__()
        self.crit = crit
        
    def forward(self, output, target):
        if not len(target.size()) == 2: return self.crit(output, target).mean()
        loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())
        return (loss1 * target[:,2] + loss2 * (1-target[:,2])).mean()
