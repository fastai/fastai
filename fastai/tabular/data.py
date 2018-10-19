"Data loading pipeline for structured data support. Loads from pandas DataFrame"
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..basic_train import *
from .models import *
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

__all__ = ['TabularDataBunch', 'TabularDataset', 'get_tabular_learner']

OptTabTfms = Optional[Collection[TabularTransform]]

def def_emb_sz(df, n, sz_dict):
    col = df[n]
    n_cat = len(col.cat.categories)+1  # extra cat for NA
    sz = sz_dict.get(n, min(50, (n_cat//2)+1))  # rule of thumb
    return n_cat,sz


class TabularDataset(DatasetBase):
    "Class for tabular data."
    def __init__(self, df:DataFrame, dep_var:str, cat_names:OptStrList=None, cont_names:OptStrList=None,
                 stats:OptStats=None, log_output:bool=False):
        if not is_numeric_dtype(df[dep_var]): df[dep_var] = df[dep_var].cat.codes.astype(np.int64)
        self.y = np2model_tensor(df[dep_var].values)
        if log_output: self.y = torch.log(self.y.float())
        self.loss_func = F.cross_entropy if self.y.dtype == torch.int64 else F.mse_loss
        n = len(self.y)
        if cat_names and len(cat_names) >= 1:
            self.cats = np.stack([c.cat.codes.values for n,c in df[cat_names].items()], 1) + 1
        else: self.cats = np.zeros((n,1))
        self.cats = LongTensor(self.cats.astype(np.int64))
        if cont_names and len(cont_names) >= 1:
            self.conts = np.stack([c.astype('float32').values for n,c in df[cont_names].items()], 1)
            means, stds = stats if stats is not None else (self.conts.mean(0), self.conts.std(0))
            self.conts = (self.conts - means[None]) / (stds[None]+1e-7)
            self.stats = means,stds
        else:
            self.conts = np.zeros((n,1), dtype=np.float32)
            self.stats = None
        self.conts = FloatTensor(self.conts)
        self.df = df

    def __len__(self)->int: return len(self.y)
    def __getitem__(self, idx)->Tuple[Tuple[LongTensor,FloatTensor], Tensor]:
        return ((self.cats[idx], self.conts[idx]), self.y[idx])
    @property
    def c(self)->int: return 1 if isinstance(self.y, FloatTensor) else self.y.max().item()+1

    def get_emb_szs(self, sz_dict): return [def_emb_sz(self.df, n, sz_dict) for n in self.cat_names]

    @classmethod
    def from_dataframe(cls, df:DataFrame, dep_var:str, tfms:OptTabTfms=None, cat_names:OptStrList=None,
                       cont_names:OptStrList=None, stats:OptStats=None, log_output:bool=False)->'TabularDataset':
        "Create a tabular dataframe from df after applying optional transforms."
        if cat_names is None: cat_names = [n for n in df.columns if is_categorical_dtype(df[n])]
        if cont_names is None: cont_names = [n for n in df.columns if is_numeric_dtype(df[n]) and not n==dep_var]
        if tfms is None: tfms = []
        for i,tfm in enumerate(tfms):
            if isinstance(tfm, TabularTransform): tfm(df, test=True)
            else:
                tfm = tfm(cat_names, cont_names)
                tfm(df)
                tfms[i] = tfm
                cat_names, cont_names = tfm.cat_names, tfm.cont_names
        ds = cls(df, dep_var, cat_names, cont_names, stats, log_output)
        ds.tfms,ds.cat_names,ds.cont_names = tfms,cat_names,cont_names
        return ds

class TabularDataBunch(DataBunch):
    "Create a `DataBunch` suitable for tabular data."
    @classmethod
    def from_df(cls, path, train_df:DataFrame, valid_df:DataFrame, dep_var:str, test_df:OptDataFrame=None,
                        tfms:OptTabTfms=None, cat_names:OptStrList=None, cont_names:OptStrList=None,
                        stats:OptStats=None, log_output:bool=False, **kwargs)->DataBunch:
        "Create a `DataBunch` from train/valid/test dataframes."
        cat_names = ifnone(cat_names, [])
        cont_names = ifnone(cont_names, list(set(train_df)-set(cat_names)-{dep_var}))
        train_ds = TabularDataset.from_dataframe(train_df, dep_var, tfms, cat_names, cont_names, stats, log_output)
        valid_ds = TabularDataset.from_dataframe(valid_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                             train_ds.cont_names, train_ds.stats, log_output)
        datasets = [train_ds, valid_ds]
        if test_df is not None:
            datasets.append(TabularDataset.from_dataframe(test_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                                      train_ds.cont_names, train_ds.stats, log_output))
        return cls.create(*datasets, path=path, **kwargs)

def get_tabular_learner(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range)
    return Learner(data, model, metrics=metrics, **kwargs)

