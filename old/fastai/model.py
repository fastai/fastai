# -*- coding: utf-8 -*-
from .imports import *
from .torch_imports import *
from .core import *
from .layer_optimizer import *
from .swa import *
from .fp16 import *

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def predict_to_bcolz(m, gen, arr, workers=4):
    arr.trim(len(arr))
    lock=threading.Lock()
    m.eval()
    for x,*_ in tqdm(gen):
        y = to_np(m(VV(x)).data)
        with lock:
            arr.append(y)
            arr.flush()

def num_features(m):
    c=children(m)
    if len(c)==0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

def torch_item(x): return x.item() if hasattr(x,'item') else x[0]

class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        xtra = []
        output = self.m(*xs)
        if isinstance(output,tuple): output,*xtra = output
        if self.fp16: self.m.zero_grad()
        else: self.opt.zero_grad() 
        loss = raw_loss = self.crit(output, y)
        if self.loss_scale != 1: assert(self.fp16); loss = loss*self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:   # Gradient clipping
            if IS_TORCH_04: nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else:           nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0: 
            #Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        if self.fp16: 
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds,tuple): preds=preds[0]
        return preds, self.crit(preds, y)

def set_train_mode(m):
    if (hasattr(m, 'running_mean') and (getattr(m,'bn_freeze',False)
              or not getattr(m,'trainable',False))): m.eval()
    elif (getattr(m,'drop_freeze',False) and hasattr(m, 'p')
          and ('drop' in type(m).__name__.lower())): m.eval()
    else: m.train()

def fit(model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
        swa_model=None, swa_start=None, swa_eval_freq=None, visualize=False, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses (can be a list)
       opts: an optimizer. Example: optim.Adam. 
       If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
       n_epochs(int or list): number of epochs (or list of number of epochs)
       crit: loss function to optimize. Example: F.cross_entropy
    """

    seq_first = kwargs.pop('seq_first', False)
    all_val = kwargs.pop('all_val', False)
    get_ep_vals = kwargs.pop('get_ep_vals', False)
    validate_skip = kwargs.pop('validate_skip', 0)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = "{!s:10} " * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs=[n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, opt.opt if hasattr(opt,'opt') else opt, crit, **kwargs)
    ep_vals = collections.OrderedDict()
    tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
    cnt_phases = np.array([ep * len(dat.trn_dl) for (ep,dat) in zip(n_epochs,data)]).cumsum()
    phase = 0
    for epoch in tnrange(tot_epochs, desc='Epoch'):
        if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch, miniters=0)
        if all_val: val_iter = IterBatch(cur_data.val_dl)

        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss, refresh=False)
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics, epoch, seq_first=seq_first, validate_skip = validate_skip)
            stop=False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            if swa_model is not None:
                if (epoch + 1) >= swa_start and ((epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
                    fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics, epoch, validate_skip = validate_skip)
                    vals += swa_vals

            if epoch > 0: 
                print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
            else:
                print(layout.format(*names))
                print_stats(epoch, [debias_loss] + vals, visualize)
            prev_val = [debias_loss] + vals
            ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
        if stop: break
    for cb in callbacks: cb.on_train_end()
    if get_ep_vals: return vals, ep_vals
    else: return vals

def append_stats(ep_vals, epoch, values, decimals=6):
    ep_vals[epoch]=list(np.round(values, decimals))
    return ep_vals

def print_stats(epoch, values, visualize, prev_val=[], decimals=6):
    layout = "{!s:^10}" + " {!s:10}" * len(values)
    values = [epoch] + list(np.round(values, decimals))
    sym = ""
    if visualize:
        if epoch == 0:                                             pass        
        elif values[1] > prev_val[0] and values[2] > prev_val[1]:  sym = " △ △"
        elif values[1] > prev_val[0] and values[2] < prev_val[1]:  sym = " △ ▼"            
        elif values[1] < prev_val[0] and values[2] > prev_val[1]:  sym = " ▼ △"            
        elif values[1] < prev_val[0] and values[2] < prev_val[1]:  sym = " ▼ ▼"
    print(layout.format(*values) + sym)

class IterBatch():
    def __init__(self, dl):
        self.idx = 0
        self.dl = dl
        self.iter = iter(dl)

    def __iter__(self): return self

    def next(self):
        res = next(self.iter)
        self.idx += 1
        if self.idx == len(self.dl):
            self.iter = iter(self.dl)
            self.idx=0
        return res

def validate_next(stepper, metrics, val_iter):
    """Computes the loss on the next minibatch of the validation set."""
    stepper.reset(False)
    with no_grad_context():
        (*x,y) = val_iter.next()
        preds,l = stepper.evaluate(VV(x), VV(y))
        res = [delistify(to_np(l))]
        res += [f(datafy(preds), datafy(y)) for f in metrics]
    stepper.reset(True)
    return res

def batch_sz(x, seq_first=False):
    if is_listy(x): x = x[0]
    return x.shape[1 if seq_first else 0]

def validate(stepper, dl, metrics, epoch, seq_first=False, validate_skip = 0):
    if epoch < validate_skip: return [float('nan')] + [float('nan')] * len(metrics)
    batch_cnts,loss,res = [],[],[]
    stepper.reset(False)
    with no_grad_context():
        t = tqdm(iter(dl), leave=False, total=len(dl), miniters=0, desc='Validation')
        for (*x,y) in t:
            y = VV(y)
            preds, l = stepper.evaluate(VV(x), y)
            batch_cnts.append(batch_sz(x, seq_first=seq_first))
            loss.append(to_np(l))
            res.append([to_np(f(datafy(preds), datafy(y))) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)[0]] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def get_prediction(x):
    if is_listy(x): x=x[0]
    return x.data

def predict(m, dl):
    preda,_ = predict_with_targs_(m, dl)
    return np.concatenate(preda)

def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))

def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x,y in iter(dl): res.append([get_prediction(to_np(m(*VV(x)))),to_np(y)])
    return zip(*res)

def predict_with_targs(m, dl):
    preda,targa = predict_with_targs_(m, dl)
    return np.concatenate(preda), np.concatenate(targa)

# From https://github.com/ncullen93/torchsample
def model_summary(m, inputs):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if is_listy(output):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
           not isinstance(module, nn.ModuleList) and
           not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)
    xs = [to_gpu(Variable(x)) for x in inputs]
    m(*xs)

    for h in hooks: h.remove()
    return summary
