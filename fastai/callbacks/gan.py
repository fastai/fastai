from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback
from ..vision.models.gan import WasserteinLoss

__all__ = ['GANTrainer', 'create_noise', 'first_disc_iter', 'standard_disc_iter']

def create_noise(x, b, noise_sz, grad=True): return x.new(b, noise_sz, 1, 1).normal_(0, 1).requires_grad_(grad)

def first_disc_iter(gen_iter):
    return 100 if (gen_iter < 25 or gen_iter%500 == 0) else 5

def standard_disc_iter(gen_iter):
    return 100 if gen_iter%500 == 0 else 5

@dataclass
class GANTrainer(LearnerCallback):
    loss_fn:LossFunction = WasserteinLoss()
    n_disc_iter:Callable = standard_disc_iter
    clip:float = 0.01
    bs:int = 64
    noise_sz:int=100
    
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
    
    def on_batch_begin(self, **kwargs):
        for p in self.learn.model.discriminator.parameters(): 
            p.data.clamp_(-self.clip, self.clip)
        
    def on_backward_begin(self, last_output, last_input, **kwargs):
        fake = self.learn.model(create_noise(last_input, last_input.size(0), self.noise_sz, False), gen=True)
        fake.requires_grad_(True)
        loss = self.loss_fn(last_output, self.learn.model(fake))
        self.dlosses.append(loss.detach().cpu())
        return loss
    
    def on_batch_end(self, last_input, **kwargs):
        self.disc_iters += 1
        if self.disc_iters == self.n_disc_iter(self.gen_iters):
            self.disc_iters = 0
            self._set_trainable(True)
            loss = self.learn.model(self.learn.model(create_noise(last_input,self.bs,self.noise_sz), gen=True)).mean().view(1)[0]
            self.glosses.append(loss.detach().cpu())
            self.learn.model.generator.zero_grad()
            loss.backward()
            self.opt_gen.step()
            self.gen_iters += 1
            self._set_trainable()