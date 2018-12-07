from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['CycleGANTrainer', 'GANTrainer', 'FixedGANSwitcher', 'AdaptiveGANSwitcher']

class GANTrainer(LearnerCallback):
    "Handles GAN Training."
    _order=-20
    def __init__(self, learn:Learner, switch_eval:bool=False, clip:float=None, beta:float=0.98, gen_first:bool=False):
        super().__init__(learn)
        self.switch_eval,self.clip,self.beta,self.gen_first = switch_eval,clip,beta,gen_first
        self.generator,self.critic = self.model.generator,self.model.critic

    def _set_trainable(self):
        train_model = self.generator if     self.gen_mode else self.critic
        loss_model  = self.generator if not self.gen_mode else self.critic
        requires_grad(train_model, True)
        requires_grad(loss_model, False)
        if self.switch_eval:
            train_model.train()
            loss_model.eval()
    
    def on_train_begin(self, **kwargs):
        "Create the optimizers for the generator and disciminator if necessary."
        if not getattr(self,'opt_gen',None):
            self.opt_gen = self.opt.new([nn.Sequential(*flatten_model(self.generator))])
        else: self.opt_gen.lr,self.opt_gen.wd = self.opt.lr,self.opt.wd
        if not getattr(self,'opt_critic',None):
            self.opt_critic = self.opt.new([nn.Sequential(*flatten_model(self.critic))])
        else: self.opt_critic.lr,self.opt_critic.wd = self.opt.lr,self.opt.wd
        self.gen_mode = self.gen_first
        self.switch(self.gen_mode)
        self.closses,self.glosses = [],[]
        self.smoothenerG,self.smoothenerC = SmoothenValue(self.beta),SmoothenValue(self.beta)
        self.recorder.no_val=True
        self.recorder.add_metric_names(['gen_loss', 'disc_loss'])
        self.imgs,self.titles = [],[]
    
    def on_train_end(self, **kwargs): 
        "Switch in generator mode for showing results."
        self.switch(gen_mode=True)
        
    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Clamp the weights with `self.clip` if it's not None."
        if self.clip is not None:
            for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
        return (last_input,last_target) if self.gen_mode else (last_target, last_input)
        
    def on_backward_begin(self, last_loss, last_output, **kwargs):
        "Record `last_loss` in the proper list."
        last_loss = last_loss.detach().cpu()
        if self.gen_mode:
            self.smoothenerG.add_value(last_loss)
            self.glosses.append(self.smoothenerG.smooth)
            self.last_gen = last_output.detach().cpu()
        else:
            self.smoothenerC.add_value(last_loss)
            self.closses.append(self.smoothenerC.smooth)
    
    def on_epoch_begin(self, epoch, **kwargs):
        "Put the critic or the generator back to eval if necessary."
        self.switch(self.gen_mode)

    def on_epoch_end(self, pbar, epoch, **kwargs):
        "Put the various losses in the recorder and show a sample image."
        self.recorder.add_metrics([getattr(self.smoothenerG,'smooth',None),getattr(self.smoothenerC,'smooth',None)])
        #if hasattr(self, 'last_gen'):
        #    self.imgs.append(Image(self.last_gen[0]/2 + 0.5))
        #    self.titles.append(f'Epoch {epoch}')
        #    pbar.show_imgs(self.imgs, self.titles)
    
    def switch(self, gen_mode:bool=None):
        "Switch the model, if `gen_mode` is provided, in the desired mode."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
        self.opt.opt = self.opt_gen.opt if self.gen_mode else self.opt_critic.opt
        self._set_trainable()
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)

@dataclass
class FixedGANSwitcher(LearnerCallback):
    "Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator."
    n_crit:Union[int,Callable]=1
    n_gen:Union[int,Callable]=1
    
    def on_train_begin(self, **kwargs):
        "Initiate the iteration counts."
        self.n_c,self.n_g = 0,0
    
    def on_batch_end(self, iteration, **kwargs):
        "Switch the model if necessary."
        if self.learn.gan_trainer.gen_mode: 
            self.n_g += 1
            n_iter,n_in,n_out = self.n_gen,self.n_c,self.n_g
        else:
            self.n_c += 1
            n_iter,n_in,n_out = self.n_crit,self.n_g,self.n_c
        target = n_iter if isinstance(n_iter, int) else n_iter(n_in)
        if target == n_out: 
            self.learn.gan_trainer.switch()
            self.n_c,self.n_g = 0,0

class AdaptiveGANSwitcher(LearnerCallback):
    "Switcher that goes back to generator/discriminator when the loes goes below `gen_thresh`\`crit_thresh`." 
    gen_thresh:float=None
    critic_thresh:float=None
    
    def on_batch_end(self, last_loss, **kwargs):
        "Switch the model if necessary."
        if self.gan_trainer.gen_mode: 
            if self.gen_thresh  is not None and last_loss < self.gen_thresh:    self.gan_trainer.switch()
        else:
            if self.crit_thresh is not None and last_loss < self.critic_thresh: self.gan_trainer.switch()
            
class CycleGANTrainer(LearnerCallback):
    "`LearnerCallback` that handles cycleGAN Training."
    _order=-20
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
        "Create the various optimizers."
        self.G_A,self.G_B = self.learn.model.G_A,self.learn.model.G_B
        self.D_A,self.D_B = self.learn.model.D_A,self.learn.model.D_B
        self.crit = self.learn.loss_func.crit
        self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G_A), *flatten_model(self.G_B))])
        self.opt_D_A = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_A))])
        self.opt_D_B = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_B))])
        self.learn.opt.opt = self.opt_G.opt
        self._set_trainable()
        self.names = ['idt_loss', 'gen_loss', 'cyc_loss', 'da_loss', 'db_loss']
        self.learn.recorder.no_val=True
        self.learn.recorder.add_metric_names(self.names)
        self.smootheners = {n:SmoothenValue(0.98) for n in self.names}

    def on_batch_begin(self, last_input, **kwargs):
        "Register the `last_input` in the loss function."
        self.learn.loss_func.set_input(last_input)

    def on_batch_end(self, last_input, last_output, **kwargs):
        "Steps through the generators then each of the critics."
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
        metrics = self.learn.loss_func.metrics + [loss_D_A, loss_D_B]
        for n,m in zip(self.names,metrics): self.smootheners[n].add_value(m)

    def on_epoch_end(self, **kwargs):
        "Put the various losses in the recorder."
        self.learn.recorder.add_metrics([s.smooth for k,s in self.smootheners.items()])
