"Data loading pipeline for structured data support. Loads from pandas `DataFrame`"
from ..torch_core import *
from .transform import *
from ..data import *
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

__all__ = ['TabularDataset', 'tabular_data_from_df']

OptTabTfms = Optional[Collection[TabularTransform]]

class TabularDataset(DatasetBase):
    "Class for tabular data."
    def __init__(self, df:DataFrame, dep_var:str, cat_names:OptStrList=None, cont_names:OptStrList=None,
                 stats:OptStats=None, log_output:bool=False):
        if not is_numeric_dtype(df[dep_var]): df[dep_var] = df[dep_var].cat.codes
        self.y = torch.tensor(df[dep_var].values).float()
        if log_output: self.y = torch.log(self.y.float())
        n = len(self.y)
        if cat_names and len(cat_names) >= 1:
            self.cats = np.stack([c.cat.codes.values for n,c in df[cat_names].items()], 1) + 1
        else: self.cats = np.zeros((n,1))
        self.cats = LongTensor(self.cats.astype(np.int64))
        if cont_names and len(cont_names) >= 1:
            self.conts = np.stack([c.astype('float32').values for n,c in df[cont_names].items()], 1)
            means, stds = stats if stats is not None else (self.conts.mean(0), self.conts.std(0))
            self.conts = (self.conts - means[None]) / stds[None]
            self.stats = means,stds
        else:
            self.conts = np.zeros((n,1), dtype=np.float32)
            self.stats = None
        self.conts = FloatTensor(self.conts)

    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, idx) -> Tuple[Tuple[LongTensor,FloatTensor], Tensor]:
        return ((self.cats[idx], self.conts[idx]), self.y[idx])
    @property
    def c(self) -> int: return 1


    @classmethod
    def from_dataframe(cls, df:DataFrame, dep_var:str, tfms:OptTabTfms=None, cat_names:OptStrList=None,
                       cont_names:OptStrList=None, stats:OptStats=None, log_output:bool=False) -> 'TabularDataset':
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

def tabular_data_from_df(path, train_df:DataFrame, valid_df:DataFrame, dep_var:str, test_df:OptDataFrame=None,
                        tfms:OptTabTfms=None, cat_names:OptStrList=None, cont_names:OptStrList=None,
                        stats:OptStats=None, log_output:bool=False, **kwargs) -> DataBunch:
    "Create a `DataBunch` from train/valid/test dataframes."
    train_ds = TabularDataset.from_dataframe(train_df, dep_var, tfms, cat_names, cont_names, stats, log_output)
    valid_ds = TabularDataset.from_dataframe(valid_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                             train_ds.cont_names, train_ds.stats, log_output)
    datasets = [train_ds, valid_ds]
    if test_df:
        datasets.appendTabularDataset.from_dataframe(valid_df, dep_var, train_ds.tfms, train_ds.cat_names,
                                                     train_ds.cont_names, train_ds.stats, log_output)
    return DataBunch.create(*datasets, path=path, **kwargs)