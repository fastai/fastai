from .imports import *
from .torch_imports import *
from .core import *
from .layer_optimizer import *

def cut_model(m, cut): return list(m.children())[:cut]

def predict_to_bcolz(m, gen, arr, workers=4):
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


class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.reset(True)

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'): self.m.reset()

    def step(self, xs, y):
        xtra = []
        output = self.m(*xs)
        if isinstance(output,(tuple,list)): output,*xtra = output
        self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.clip:   # Gradient clipping
            nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        return raw_loss.data[0]

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds,(tuple,list)): preds=preds[0]
        return preds, self.crit(preds,y)


def set_train_mode(m):
    if hasattr(m, 'running_mean') and not (hasattr(m,'trainable') and m.trainable): m.eval()
    else: m.train()


def fit(model, data, epochs, opt, crit, metrics=None, callbacks=None, **kwargs):
    """ Fits a model

    Arguments:
       model (model):example:
           net = nn.Sequential(
               nn.Linear(28*28, 256),
               nn.ReLU(),
               nn.Linear(256, 10)
           )
           net = to_gpu(net)
       data (DataModel): see examples of DataModel
           it data loaders: data.trn_dl and data.val_dl
       opt: optimization. Example: opt=optim.Adam(net.parameters())
       epochs(int): number of epochs
       crit: loss function to optimize. Example: F.cross_entropy
    """
    stepper = Stepper(model, opt, crit, **kwargs)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.

    for epoch in tnrange(epochs, desc='Epoch'):
        stepper.reset(True)
        t = tqdm(iter(data.trn_dl), leave=False)
        for (*x,y) in t:
            batch_num += 1
            loss = stepper.step(V(x),V(y))
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(train_loss=debias_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(debias_loss)
            if stop: return

        vals = validate(stepper, data.val_dl, metrics)
        print("epoch: {:4d}, train_loss: {:10.6f}, val_loss: {:10.6f}, val_acc: {:10.6f}".\
              format(epoch, avg_loss, vals[0], vals[1]))
        stop=False
        for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
        if stop: break

def validate(stepper, dl, metrics):
    loss,res = [],[]
    stepper.reset(False)
    for (*x,y) in iter(dl):
        preds,l = stepper.evaluate(VV(x), VV(y))
        loss.append(to_np(l))
        res.append([f(to_np(preds),to_np(y)) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))

def predict(m, dl): return predict_with_targs(m, dl)[0]

def get_prediction(x):
    if isinstance(x,(tuple,list)): x=x[0]
    return x.data

def predict_with_targs(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    preda,targa = zip(*[(get_prediction(m(*VV(x))),y)
                        for *x,y in iter(dl)])
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))

