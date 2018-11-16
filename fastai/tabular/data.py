"Data loading pipeline for structured data support. Loads from pandas DataFrame"
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *
from ..basic_train import *
from .models import *
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

__all__ = ['TabularDataBunch', 'TabularLine', 'TabularList', 'TabularProcessor', 'tabular_learner']

OptTabTfms = Optional[Collection[TabularProc]]

def emb_sz_rule(n_cat:int)->int: return min(50, (n_cat//2)+1)
#def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))

def def_emb_sz(df, n, sz_dict):
    col = df[n]
    n_cat = len(col.cat.categories)+1  # extra cat for NA
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz

def _text2html_table(items:Collection[Collection[str]], widths:Collection[int])->str:
    html_code = f"<table>"
    for w in widths: html_code += f"  <col width='{w}px'>"
    for line in items:
        html_code += "  <tr>\n"
        html_code += "\n".join([f"    <th>{o}</th>" for o in line if len(o) >= 1])
        html_code += "\n  </tr>\n"
    return html_code + "</table>\n"

#def _fix_empty_tensor(o): = [tensor(cats) if len(cats) != 0 else tensor(0),

class TabularLine(ItemBase):
    def __init__(self, cats, conts, classes, names):
        self.cats,self.conts,self.classes,self.names = cats,conts,classes,names
        self.data = [tensor(cats), tensor(conts)]

    def __str__(self):
        res = ''
        for c, n in zip(self.cats, self.names[:len(self.cats)]):
            res += f"{n} {(self.classes[n][c-1] if c != 0 else 'nan')}; "
        for c,n in zip(self.conts, self.names[len(self.cats):]):
            res += f'{n} {c:.4f}; '
        return res

    def show_batch(self, idxs:Collection[int], rows:int, ds:Dataset, **kwargs)->None:
        "Show the data in `idxs` on a few `rows` from `ds`."
        from IPython.display import display, HTML
        x,y = ds[0]
        items = [x.names + ['target']]
        for i in idxs[:rows]:
            x,y = ds[i]
            res = []
            for c, n in zip(x.cats, self.names[:len(x.cats)]):
                res.append(str(x.classes[n][c-1]) if c != 0 else 'nan')
            res += [f'{c:.4f}' for c in x.conts] + [str(y)]
            items.append(res)
        display(HTML(_text2html_table(items, [10] * len(items[0]))))

class TabularProcessor(PreProcessor):
    def __init__(self, ds:ItemBase=None, procs=None): 
        procs = ifnone(procs, ds.procs if ds is not None else None)
        self.procs = listify(procs)

    def process_one(self, item):
        df = pd.DataFrame([item,item])
        for proc in self.procs: proc(df, test=True)
        if self.cat_names is not None:
            codes = np.stack([c.cat.codes.values for n,c in df[self.cat_names].items()], 1).astype(np.int64) + 1
        else: codes = [[]]
        if self.cont_names is not None:
            conts = np.stack([c.astype('float32').values for n,c in df[self.cont_names].items()], 1)
        else: conts = [[]]
        classes = None
        col_names = list(df[self.cat_names].columns.values) + list(df[self.cont_names].columns.values)
        return TabularLine(codes[0], conts[0], classes, col_names)

    def process(self, ds):
        for i,proc in enumerate(self.procs):
            if isinstance(proc, TabularProc): proc(ds.xtra, test=True)
            else:
                #cat and cont names may have been changed by transform (like Fill_NA)
                proc = proc(ds.cat_names, ds.cont_names)
                proc(ds.xtra)
                ds.cat_names,ds.cont_names = proc.cat_names,proc.cont_names
                self.procs[i] = proc
        self.cat_names,self.cont_names = ds.cat_names,ds.cont_names
        if len(ds.cat_names) != 0:
            ds.codes = np.stack([c.cat.codes.values for n,c in ds.xtra[ds.cat_names].items()], 1).astype(np.int64) + 1
            ds.classes = OrderedDict({n:c.cat.categories.values
                                      for n,c in ds.xtra[ds.cat_names].items()})
            cat_cols = list(ds.xtra[ds.cat_names].columns.values)
        else: ds.codes,ds.classes,cat_cols = None,None,[]
        if len(ds.cont_names) != 0:
            ds.conts = np.stack([c.astype('float32').values for n,c in ds.xtra[ds.cont_names].items()], 1)
            cont_cols = list(ds.xtra[ds.cont_names].columns.values)
        else: ds.conts,cont_cols = None,[]
        ds.col_names = cat_cols + cont_cols
        
class TabularList(ItemList):
    _item_cls=TabularLine
    _processor=TabularProcessor
    def __init__(self, items:Iterator, cat_names:OptStrList=None, cont_names:OptStrList=None,
                 procs=None, **kwargs)->'TabularList':
        #dataframe is in xtra, items is just a range of index
        if cat_names is None:  cat_names = []
        if cont_names is None: cont_names = []
        super().__init__(range_of(items), **kwargs)
        self.cat_names,self.cont_names,self.procs = cat_names,cont_names,procs

    @classmethod
    def from_df(cls, df:DataFrame, cat_names:OptStrList=None, cont_names:OptStrList=None, procs=None, **kwargs)->'ItemList':
        "Get the list of inputs in the `col` of `path/csv_name`."
        return cls(items=range(len(df)), cat_names=cat_names, cont_names=cont_names, procs=procs, xtra=df, **kwargs)

    def new(self, items:Iterator, **kwargs)->'TabularList':
        return super().new(items=items, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs, **kwargs)

    def get(self, o):
        codes = [] if self.codes is None else self.codes[o]
        conts = [] if self.conts is None else self.conts[o]
        return self._item_cls(codes, conts, self.classes, self.col_names)

    def get_emb_szs(self, sz_dict):
        "Return the default embedding sizes suitable for this data or takes the ones in `sz_dict`."
        return [def_emb_sz(self.xtra, n, sz_dict) for n in self.cat_names]

class TabularDataBunch(DataBunch):
    "Create a `DataBunch` suitable for tabular data."
    @classmethod
    def from_df(cls, path, df:DataFrame, dep_var:str, valid_idx:Collection[int], procs:OptTabTfms=None,
                cat_names:OptStrList=None, cont_names:OptStrList=None, classes:Collection=None, **kwargs)->DataBunch:
        "Create a `DataBunch` from train/valid/test dataframes."
        cat_names = ifnone(cat_names, [])
        cont_names = ifnone(cont_names, list(set(df)-set(cat_names)-{dep_var}))
        procs = listify(procs)
        return (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(valid_idx)
                           .label_from_df(cols=dep_var, classes=None)
                           .databunch())

def tabular_learner(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range, use_bn=use_bn)
    return Learner(data, model, metrics=metrics, **kwargs)

