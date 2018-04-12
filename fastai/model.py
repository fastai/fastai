from .imports import *
from .torch_imports import *
from .core import *
from .layer_optimizer import *
from .fp16 import *

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
            nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
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


def fit(model, data, epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses
       opt: optimizer. Example: opt=optim.Adam(net.parameters())
       epochs(int): number of epochs
       crit: loss function to optimize. Example: F.cross_entropy
    """
    all_val = kwargs.pop('all_val') if 'all_val' in kwargs else False
    sampler = kwargs.pop('sampler') if 'sampler' in kwargs else None
    stepper = stepper(model, opt, crit, **kwargs)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    layout = "{!s:10} " * len(names)

    num_batch = len(data.trn_dl)
    if epochs<1:
        num_batch = int(num_batch*epochs)
        epochs = 1

    for epoch in tnrange(epochs, desc='Epoch'):
        if sampler: sampler.set_epoch(epoch)
        stepper.reset(True)
        t = tqdm(iter(data.trn_dl), leave=False, total=num_batch)
        i = 0
        if all_val: val_iter = IterBatch(data.val_dl)
        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss)
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if i>num_batch: break
            i += 1

        if not all_val:
            vals = validate(stepper, data.val_dl, metrics)
            if epoch == 0: print(layout.format(*names))
            print_stats(epoch, [debias_loss] + vals)
            stop=False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
        if stop: break

    for cb in callbacks: cb.on_train_end()
    return vals


def print_stats(epoch, values, decimals=6):
    layout = "{!s:^10}" + " {!s:10}" * len(values)
    values = [epoch] + list(np.round(values, decimals))
    print(layout.format(*values))

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
    (*x,y) = val_iter.next()
    preds,l = stepper.evaluate(VV(x), VV(y))
    res = [to_np(l)[0]]
    res += [f(preds.data,y) for f in metrics]
    stepper.reset(True)
    return res

def validate(stepper, dl, metrics):
    batch_cnts,loss,res = [],[],[]
    stepper.reset(False)
    for (*x,y) in iter(dl):
        y = VV(y)
        preds,l = stepper.evaluate(VV(x), y)
        if isinstance(x,list): batch_cnts.append(len(x[0]))
        else: batch_cnts.append(len(x))
        loss.append(to_np(l))
        res.append([f(preds.data,y.data) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def get_prediction(x):
    if is_listy(x): x=x[0]
    return x.data

def predict(m, dl):
    preda,_ = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda))

def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))

def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x,y in iter(dl): res.append([get_prediction(m(*VV(x))),y])
    return zip(*res)

def predict_with_targs(m, dl):
    preda,targa = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))

# From https://github.com/ncullen93/torchsample
def model_summary(m, input_size):
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

    if is_listy(input_size[0]):
        x = [to_gpu(Variable(torch.rand(3,*in_size))) for in_size in input_size]
    else: x = [to_gpu(Variable(torch.rand(3,*input_size)))]
    m(*x)

    for h in hooks: h.remove()
    return summary

