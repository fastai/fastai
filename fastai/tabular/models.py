from ..torch_core import *
from ..layers import *
from ..basic_train import *
from ..basic_data import *

__all__ = ['TabularModel','ClassificationInterpretationTabular']

class TabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


class ClassificationInterpretationTabular():
    "Interpretation methods for classification models with tabular data."
    def __init__(self, data:DataBunch, probs:Tensor, y_true:Tensor, losses:Tensor):
        self.data,self.probs,self.y_true,self.losses = data,probs,y_true,losses
        self.pred_class = self.probs.argmax(dim=1)

    @classmethod
    def from_learner(cls, learn:Learner, ds_type:DatasetType=DatasetType.Valid):
        "Create an instance of `ClassificationInterpretationTabular`"
        preds = learn.get_preds(ds_type=ds_type, with_loss=True)
        return cls(learn.data, *preds)

    def confusion_matrix(self):
        "Confusion matrix as an `np.ndarray`."
        x=torch.arange(0,self.data.c)
        cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
        for i in range(0, self.y_true.shape[0], 1):
            cm_slice = ((self.pred_class[i:i+1]==x[:,None])
                        & (self.y_true[i:i+1]==x[:,None,None])).sum(2)
            torch.add(cm, cm_slice, out=cm)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", norm_dec:int=2, **kwargs)->None:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix()
        plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, list(range(self.data.c)), rotation=90)
        plt.yticks(tick_marks, list(range(self.data.c)), rotation=0)

        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
