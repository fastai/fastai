from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback
from ..vision.models.gan import WasserteinLoss

__all__ = ['CycleGANTrainer', 'GANTrainer', 'create_noise', 'first_disc_iter', 'standard_disc_iter']

def create_noise(x, b, noise_sz, grad=True): return x.new(b, noise_sz, 1, 1).normal_(0, 1).requires_grad_(grad)

def first_disc_iter(gen_iter):
    return 100 if (gen_iter < 25 or gen_iter%500 == 0) else 5

def standard_disc_iter(gen_iter):
    return 100 if gen_iter%500 == 0 else 5

@dataclass
class GANTrainer(LearnerCallback):
    _order=-20
    loss_func:LossFunction = WasserteinLoss()
    n_disc_iter:Callable = standard_disc_iter
    clip:float = 0.01
    bs:int = 64
    noise_sz:int=100
    beta:float=0.98
    
    def _set_trainable(self, gen=False):
        requires_grad(self.learn.model.generator, gen)
        requires_grad(self.learn.model.discriminator, not gen)
        if gen:
            self.opt_gen.lr, self.opt_gen.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_gen.wd, self.opt_gen.beta = self.learn.opt.wd, self.learn.opt.beta
    
    def on_train_begin(self, **kwargs):
        self.opt_gen = self.learn.opt.new([nn.Sequential(*flatten_model(self.learn.model.generator))])
        self.opt_disc = self.learn.opt.new([nn.Sequential(*flatten_model(self.learn.model.discriminator))])
        self.learn.opt.opt = self.opt_disc.opt
        self.disc_iters, self.gen_iters = 0, 0
        self._set_trainable()
        self.dlosses,self.glosses = [],[]
        self.smoothenerG,self.smoothenerD = SmoothenValue(self.beta),SmoothenValue(self.beta)
        self.learn.recorder.add_metric_names(['gen_loss', 'disc_loss'])
    
    def on_batch_begin(self, **kwargs):
        for p in self.learn.model.discriminator.parameters(): 
            p.data.clamp_(-self.clip, self.clip)
        
    def on_backward_begin(self, last_output, last_input, **kwargs):
        fake = self.learn.model(create_noise(last_input, last_input.size(0), self.noise_sz, False), gen=True)
        fake.requires_grad_(True)
        loss = self.loss_func(last_output, self.learn.model(fake))
        self.smoothenerD.add_value(loss.detach().cpu())
        self.dlosses.append(self.smoothenerD.smooth)
        return loss
    
    def on_batch_end(self, last_input, **kwargs):
        self.disc_iters += 1
        if self.disc_iters == self.n_disc_iter(self.gen_iters):
            self.disc_iters = 0
            self._set_trainable(True)
            loss = self.learn.model(self.learn.model(create_noise(last_input,self.bs,self.noise_sz), gen=True)).mean().view(1)[0]
            self.smoothenerG.add_value(loss.detach().cpu())
            self.glosses.append(self.smoothenerG.smooth)
            self.learn.model.generator.zero_grad()
            loss.backward()
            self.opt_gen.step()
            self.gen_iters += 1
            self._set_trainable()
    
    def on_epoch_end(self, **kwargs):
        self.learn.recorder.add_metrics([self.smoothenerG.smooth,self.smoothenerD.smooth])
        
class CycleGANTrainer(LearnerCallback):
    
    def _set_trainable(self, D_A=False, D_B=False):
        gen = (not D_A) and (not D_B)
        requires_grad(self.learn.model.G_A, gen)
        requires_grad(self.learn.model.G_B, gen)
        requires_grad(self.learn.model.D_A, D_A)
        requires_grad(self.learn.model.D_B, D_B)
        if not gen:
            self.opt_D_A.lr, self.opt_D_A.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_A.wd, self.opt_D_A.beta = self.learn.opt.wd, self.learn.opt.beta
            self.opt_D_B.lr, self.opt_D_B.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_B.wd, self.opt_D_B.beta = self.learn.opt.wd, self.learn.opt.beta
    
    def on_train_begin(self, **kwargs):
        self.G_A,self.G_B = self.learn.model.G_A,self.learn.model.G_B
        self.D_A,self.D_B = self.learn.model.D_A,self.learn.model.D_B
        self.crit = self.learn.loss_func.crit
        self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G_A), *flatten_model(self.G_B))])
        self.opt_D_A = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_A))])
        self.opt_D_B = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_B))])
        self.learn.opt.opt = self.opt_G.opt
        self._set_trainable()
        
    def on_batch_begin(self, last_input, **kwargs):
        self.learn.loss_func.set_input(last_input)
    
    def on_batch_end(self, last_input, last_output, **kwargs):
        self.G_A.zero_grad(); self.G_B.zero_grad()
        fake_A, fake_B = last_output[0].detach(), last_output[1].detach()
        real_A, real_B = last_input
        self._set_trainable(D_A=True)
        self.D_A.zero_grad()
        loss_D_A = 0.5 * (self.crit(self.D_A(real_A), True) + self.crit(self.D_A(fake_A), False))
        loss_D_A.backward()
        self.opt_D_A.step()
        self._set_trainable(D_B=True)
        self.D_B.zero_grad()
        loss_D_B = 0.5 * (self.crit(self.D_B(real_B), True) + self.crit(self.D_B(fake_B), False))
        loss_D_B.backward()
        self.opt_D_B.step()
        self._set_trainable()