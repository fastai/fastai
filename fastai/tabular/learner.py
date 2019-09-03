"`Learner` support for tabular data."
from ..torch_core import *
from .transform import *
from .data import *
from .models import *
from ..basic_data import *
from ..basic_train import *
from ..train import ClassificationInterpretation

__all__ = ['tabular_learner']

def tabular_learner(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **learn_kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range, use_bn=use_bn)
    return Learner(data, model, metrics=metrics, **learn_kwargs)

@classmethod
def _cl_int_from_learner(cls, learn:Learner, ds_type=DatasetType.Valid, activ:nn.Module=None):
    "Creates an instance of 'ClassificationInterpretation"
    preds = learn.get_preds(ds_type=ds_type, activ=activ, with_loss=True)
    return cls(learn, *preds, ds_type=ds_type)

def _cl_int_plot_tab_top_losses(self, k, largest:bool=True, return_table:bool=False)->Optional[plt.Figure]:
    "Generates a dataframe of 'top_losses' along with their prediction, actual, loss, and probability of the actual class."
    tl_val, tl_idx = self.top_losses(k, largest)
    classes = self.data.classes
    cat_names = self.data.x.cat_names
    cont_names = self.data.x.cont_names
    df = pd.DataFrame(columns=[['Prediction', 'Actual', 'Loss', 'Probability'] + cat_names + cont_names])
    for i, idx in enumerate(tl_idx):
        da, cl = self.data.dl(self.ds_type).dataset[idx]
        cl = int(cl)
        t1 = str(da)
        t1 = t1.split(';')
        arr = []
        arr.extend([classes[self.pred_class[idx]], classes[cl], f'{self.losses[idx]:.2f}',
                    f'{self.preds[idx][cl]:.2f}'])
        for x in range(len(t1)-1):
            _, value = t1[x].rsplit(' ', 1)
            arr.append(value)
        df.loc[i] = arr
    display(df)
    return_fig = return_table
    if ifnone(return_fig, defaults.return_fig): return df


ClassificationInterpretation.from_learner = _cl_int_from_learner
ClassificationInterpretation.plot_tab_top_losses = _cl_int_plot_tab_top_losses

def _learner_interpret(learn:Learner, ds_type:DatasetType = DatasetType.Valid):
    "Create a 'ClassificationInterpretation' object from 'learner' on 'ds_type'."
    return ClassificationInterpretation.from_learner(learn, ds_type=ds_type)

Learner.interpret = _learner_interpret