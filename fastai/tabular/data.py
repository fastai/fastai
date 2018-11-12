"Data loading pipeline for structured data support. Loads from pandas DataFrame"
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *
from ..basic_train import *
from .models import *
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

__all__ = ['TabularDataBunch', 'TabularLine', 'TabularList', 'get_tabular_learner']

OptTabTfms = Optional[Collection[TabularTransform]]

def def_emb_sz(df, n, sz_dict):
    col = df[n]
    n_cat = len(col.cat.categories)+1  # extra cat for NA
    sz = sz_dict.get(n, min(50, (n_cat//2)+1))  # rule of thumb
    return n_cat,sz

def _text2html_table(items:Collection[Collection[str]], widths:Collection[int])->str:
    html_code = f"<table>"
    for w in widths: html_code += f"  <col width='{w}px'>"
    for line in items:
        html_code += "  <tr>\n"
        html_code += "\n".join([f"    <th>{o}</th>" for o in line if len(o) >= 1])
        html_code += "\n  </tr>\n"
    return html_code + "</table>\n"

class TabularLine(ItemBase):
    def __init__(self, cats, conts, classes, names): 
        self.cats,self.conts,self.classes,self.names = cats,conts,classes,names
        self.data = [tensor(cats), tensor(conts)]
        
    def __str__(self):  
        res = ''
        for c, n in zip(self.cats, self.names[:len(self.cats)]):
            res += f"{n} {(self.classes[n][c-1] if c != 0 else 'nan')}\n"
        for c,n in zip(self.conts, self.names[len(self.cats):]):
            res += f'{n} {c:.4f}\n'
        return res
    
    def show_batch(self, idxs:Collection[int], rows:int, ds:Dataset, figsize:Tuple[int,int]=(9,10))->None:
        from IPython.display import display, HTML
        x,y = ds[0]
        items = [x.names]
        for i in idxs[:rows]:
            x,y = ds[i]
            res = []
            for c, n in zip(x.cats, self.names[:len(x.cats)]):
                res.append(str(x.classes[n][c-1]) if c != 0 else 'nan')
            res += [f'{c:.4f}' for c in x.conts] 
            items.append(res)
        display(HTML(_text2html_table(items, [10] * len(items[0]))))

class TabularList(ItemList):
    def __init__(self, items:Iterator, cat_names:OptStrList=None, cont_names:OptStrList=None, create_func:Callable=None, 
                 path:PathOrStr='.', xtra=None):
        #dataframe is in xtra, items is just a range of index
        assert xtra is not None and len(xtra)==len(items), "Use from_df or from_csv"
        super().__init__(range(len(items)), create_func=create_func, path=path, xtra=xtra)
        self.cat_names,self.cont_names = cat_names,cont_names
    
    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', create_func:Callable=None, cat_names:OptStrList=None, 
                cont_names:OptStrList=None)->'ItemList':
        "Get the list of inputs in the `col` of `path/csv_name`."
        res = cls(create_func=create_func, items=range(len(df)), path=path, xtra=df,
                  cat_names=cat_names, cont_names=cont_names)
        return res
    
    def new(self, items:Iterator, xtra:Any=None)->'TabularList':
        return self.__class__(items=items, cat_names=self.cat_names, cont_names=self.cont_names,
                              create_func=self.create_func, path=self.path, xtra=xtra)
    
    def get(self, o): 
        return TabularLine(self.codes[o], self.conts[o], self.classes, self.col_names)
    
    def get_emb_szs(self, sz_dict): return [def_emb_sz(self.xtra, n, sz_dict) for n in self.cat_names]
    
    def preprocess(self, tfms=None):
        tfms,new_tfms = ifnone(tfms,[]),[]
        for tfm in tfms:
            if isinstance(tfm, TabularTransform): tfm(self.xtra, test=True)
            else:
                #cat and cont names may have been changed by transform (like Fill_NA)
                tfm = tfm(self.cat_names, self.cont_names)
                tfm(self.xtra)
                new_tfms.append(tfm)
                self.cat_names, self.cont_names = tfm.cat_names, tfm.cont_names
        self.codes = np.stack([c.cat.codes.values for n,c in self.xtra[self.cat_names].items()], 1).astype(np.int64) + 1
        self.conts = np.stack([c.astype('float32').values for n,c in self.xtra[self.cont_names].items()], 1)
        self.classes = {n:c.cat.categories.values for n,c in self.xtra[self.cat_names].items()}
        self.col_names = list(self.xtra[self.cat_names].columns.values) 
        self.col_names += list(self.xtra[self.cont_names].columns.values)
        self.preprocess_kwargs = {'tfms':new_tfms}
    
class TabularDataBunch(DataBunch):
    "Create a `DataBunch` suitable for tabular data."
    @classmethod
    def from_df(cls, path, df:DataFrame, dep_var:str, valid_idx:Collection[int], tfms:OptTabTfms=None, 
                cat_names:OptStrList=None, cont_names:OptStrList=None, classes:Collection=None, **kwargs)->DataBunch:
        "Create a `DataBunch` from train/valid/test dataframes."
        cat_names = ifnone(cat_names, [])
        cont_names = ifnone(cont_names, list(set(df)-set(cat_names)-{dep_var}))
        tfms = ifnone(tfms, [])
        return (TabularList.from_df(df, path, cat_names=cat_names, cont_names=cont_names)
                           .split_by_idx(valid_idx)
                           .label_from_df(cols=dep_var, classes=classes)
                           .preprocess(tfms=tfms)
                           .databunch(**kwargs))

def get_tabular_learner(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range, use_bn=use_bn)
    return Learner(data, model, metrics=metrics, **kwargs)

