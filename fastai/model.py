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


def set_train_mode(m):
    if hasattr(m, 'running_mean') and not (hasattr(m,'trainable') and m.trainable): m.eval()
    else: m.train()


def fit(stepper, data, epochs, metrics=None, callbacks=None):
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.

    for epoch in tnrange(epochs, desc='Epoch'):
        stepper.reset(True)
        t = tqdm(iter(data.trn_dl), leave=False)
        for (*x,y) in t:
        #t = trange(len(data.trn_dl), leave=False)
        #dl = iter(data.trn_dl)
        #for i in t:
            #*x,y =next(dl)
            batch_num += 1
            loss = stepper.step(V(x),V(y))
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(debias_loss)
            if stop: return

        vals = validate(stepper, data.val_dl, metrics)
        print(np.round([epoch, avg_loss] + vals, 6))
        stop=False
        for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
        if stop: return

def validate(stepper, dl, metrics):
    loss,res = [],[]
    stepper.reset(False)
    for (*x,y) in iter(dl):
    #for i in range(len(dl)):
        #(*x,y) = next(dl)
        preds,l = stepper.evaluate(VV(x), VV(y))
        loss.append(to_np(l))
        res.append([f(to_np(preds),to_np(y)) for f in metrics])
    return [np.mean(loss)] + list(np.mean(np.stack(res),0))

def predict(m, dl): return predict_with_targs(m, dl)[0]

def predict_with_targs(m, dl):
    m.eval()
    preda,targa = zip(*[(m(*VV(x)),y) for *x,y in dl])
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))

