"`Learner` support for computer vision"
from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from .image import *
from . import models
from ..callback import *
from ..layers import *
from ..callbacks.hooks import num_features_model
from ..callbacks.gan import GANTrainer, NoisyGANTrainer

__all__ = ['create_cnn', 'create_body', 'create_head', 'ClassificationInterpretation', 'unet_learner', 'GANLearner', 'gan_learner']
# By default split models between first and second layer
def _default_split(m:nn.Module): return (m[1],)
# Split a resnet style model
def _resnet_split(m:nn.Module): return (m[0][6],m[1])

_default_meta = {'cut':-1, 'split':_default_split}
_resnet_meta  = {'cut':-2, 'split':_resnet_split }

model_meta = {
    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
    models.resnet152:{**_resnet_meta}}

def cnn_config(arch):
    "Get the metadata associated with `arch`."
    torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)

def create_body(model:nn.Module, cut:Optional[int]=None, body_fn:Callable[[nn.Module],nn.Module]=None):
    "Cut off the body of a typically pretrained `model` at `cut` or as specified by `body_fn`."
    return (nn.Sequential(*list(model.children())[:cut]) if cut
            else body_fn(model) if body_fn else model)

def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,True,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def create_cnn(data:DataBunch, arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None,
                bn_final:bool=False, **kwargs:Any)->Learner:
    "Build convnet style learners."
    meta = cnn_config(arch)
    body = create_body(arch(pretrained), ifnone(cut,meta['cut']))
    nf = num_features_model(body) * 2
    head = custom_head or create_head(nf, data.c, lin_ftrs, ps=ps, bn_final=bn_final)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn

def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 **kwargs:Any)->None:
    "Build Unet learners. `kwargs` are passed down to `conv_layer`."
    meta = cnn_config(arch)
    body = create_body(arch(pretrained), meta['cut'])
    model = to_device(models.unet.DynamicUnet(body, n_classes=data.c, blur=blur, blur_final=blur_final,
                                              norm_type=norm_type), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

class ClassificationInterpretation():
    "Interpretation methods for classification models."
    def __init__(self, data:DataBunch, probs:Tensor, y_true:Tensor, losses:Tensor):
        self.data,self.probs,self.y_true,self.losses = data,probs,y_true,losses
        self.pred_class = self.probs.argmax(dim=1)

    @classmethod
    def from_learner(cls, learn:Learner, ds_type:DatasetType=DatasetType.Valid, tta=False):
        "Create an instance of `ClassificationInterpretation`. `tta` indicates if we want to use Test Time Augmentation."
        preds = learn.TTA(with_loss=True) if tta else learn.get_preds(ds_type=ds_type, with_loss=True)
        return cls(learn.data, *preds)

    def top_losses(self, k:int=None, largest=True):
        "`k` largest(/smallest) losses and indexes, defaulting to all losses (sorted by `largest`)."
        return self.losses.topk(ifnone(k, len(self.losses)), largest=largest)

    def plot_top_losses(self, k, largest=True, figsize=(12,12)):
        "Show images in `top_losses` along with their prediction, actual, loss, and probability of predicted class."
        tl_val,tl_idx = self.top_losses(k,largest)
        classes = self.data.classes
        rows = math.ceil(math.sqrt(k))
        fig,axes = plt.subplots(rows,rows,figsize=figsize)
        fig.suptitle('prediction/actual/loss/probability', weight='bold', size=14)
        for i,idx in enumerate(tl_idx):
            im,cl = self.data.valid_ds[idx]
            cl = int(cl)
            im.show(ax=axes.flat[i], title=
                f'{classes[self.pred_class[idx]]}/{classes[cl]} / {self.losses[idx]:.2f} / {self.probs[idx][cl]:.2f}')

    def confusion_matrix(self, slice_size:int=None):
        "Confusion matrix as an `np.ndarray`."
        x=torch.arange(0,self.data.c)
        if slice_size is None: cm = ((self.pred_class==x[:,None]) & (self.y_true==x[:,None,None])).sum(2)
        else:
            cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
            for i in range(0, self.y_true.shape[0], slice_size):
                cm_slice = ((self.pred_class[i:i+slice_size]==x[:,None])
                            & (self.y_true[i:i+slice_size]==x[:,None,None])).sum(2)
                torch.add(cm, cm_slice, out=cm)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", norm_dec:int=2,
                              slice_size:int=None, **kwargs)->None:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix(slice_size=slice_size)
        plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = arange_of(self.data.classes)
        plt.xticks(tick_marks, self.data.classes, rotation=90)
        plt.yticks(tick_marks, self.data.classes, rotation=0)

        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

    def most_confused(self, min_val:int=1, slice_size:int=None)->Collection[Tuple[str,str,int]]:
        "Sorted descending list of largest non-diagonal entries of confusion matrix."
        cm = self.confusion_matrix(slice_size=slice_size)
        np.fill_diagonal(cm, 0)
        res = [(self.data.classes[i],self.data.classes[j],cm[i,j])
                for i,j in zip(*np.where(cm>min_val))]
        return sorted(res, key=itemgetter(2), reverse=True)

class GANLearner(Learner):
    "`Learner` overwriting `predict` and `show_results` for GANs."
    def add_gan_trainer(self, cb):
        "Add the `GanTrainer` callback cb."
        self.gan_trainer = cb
        self.callbacks.append(cb)

    def predict(self):
        "Predict one batch of fake images."
        x,y = next(iter(self.data.train_dl))
        out = self.model(self.gan_trainer.input_fake(x, grad=False), gen=True)
        norm = getattr(self.data,'norm',False)
        if norm: out = self.data.denorm(out)
        return out.detach().cpu()

    def show_results(self, rows:int=5, figsize=(10,10)):
        "Show `rows` by `rows` fake images with `figsize`."
        out = self.predict()
        xs = [self.data.train_ds.x.reconstruct(o) for o in out[:rows*rows]]
        self.data.train_ds.x.show_xys(xs, [EmptyLabel()] * (rows*rows))

def gan_learner(data, generator, discriminator, loss_funcD=None, loss_funcG=None, noise_size:int=None, wgan:bool=False,
                **kwargs):
    "Create a `GANLearner` from `data` with a `generator` and a `discriminator`."
    gan = models.GAN(generator, discriminator)
    learn = GANLearner(data, gan, loss_func=NoopLoss(), **kwargs)
    if wgan: loss_funcD,loss_funcG = WassersteinLoss(),NoopLoss()
    if noise_size is None: cb = GANTrainer(learn, loss_funcD, loss_funcG)
    else: cb = NoisyGANTrainer(learn, loss_funcD, loss_funcG, bs=data.batch_size, noise_sz=noise_size)
    learn.add_gan_trainer(cb)
    return learn
