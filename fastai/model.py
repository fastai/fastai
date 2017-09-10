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
    if hasattr(c[-1], 'num_features'): return c[-1].num_features
    elif hasattr(c[-1], 'out_features'): return c[-1].out_features
    if hasattr(c[-2], 'num_features'): return c[-2].num_features
    elif hasattr(c[-2], 'out_features'): return c[-2].out_features
    return num_features(children(m)[-1])

def get_probabilities(net, loader):
    net.eval()
    return np.vstack(net(VV(data)) for data, *_ in loader)

def step(m, opt, xs, y, crit):
    loss = crit(m(*V(xs)), V(y))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.data[0]

def set_train_mode(m):
    if hasattr(m, 'running_mean') and not (hasattr(m,'trainable') and m.trainable): m.eval()
    else: m.train()

def fit(m, data, epochs, crit, opt, metrics=None, callbacks=None):
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98

    apply_leaf(m, set_train_mode)
    batch_num,avg_loss=0,0.
    for epoch in trange(epochs, desc='Epoch'):
        apply_leaf(m, set_train_mode)
        t = trange(len(data.trn_dl), leave=False)
        dl = iter(data.trn_dl)
        for i in t:
            batch_num += 1
            *x,y =next(dl)
            loss = step(m,opt,x,y, crit)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(debias_loss)
            if stop: return

        vals = validate(m, data.val_dl, crit, metrics)
        print(np.round([avg_loss] + vals, 6))
        stop=False
        for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
        if stop: return

def validate(m, dl, crit, metrics):
    m.eval()
    loss,res = [],[]
    for (*x,y) in dl:
        y = y.cuda()
        preds = m(*VV(x))
        loss.append(to_np(crit(preds,VV(y))))
        res.append([f(to_np(preds),to_np(y)) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))

def predict(m, dl):
    m.eval()
    return torch.cat([m(*VV(x)) for *x,_ in dl]).data.cpu()

def predict_with_targs(m, dl):
    m.eval()
    preda,targa = zip(*[(m(*VV(x)),y) for *x,y in dl])
    return torch.cat(preda).data.cpu(), torch.cat(targa)

