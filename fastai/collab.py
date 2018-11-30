"Module support for Collaborative Filtering"
from .torch_core import *
from .basic_train import *
from .basic_data import *
from .data_block import *
from .layers import *
from .tabular import *

__all__ = ['EmbeddingDotBias', 'collab_learner', 'CollabDataBunch', 'CollabLine', 'CollabList']

class CollabLine(TabularLine):
    "Base item for collaborative filtering, subclasses `TabularLine`."
    def __init__(self, cats, conts, classes, names):
        super().__init__(cats, conts, classes, names)
        self.data = [self.data[0][0],self.data[0][1]]

class CollabList(TabularList): 
    "Base `ItemList` for collaborative filtering, subclasses `TabularList`."
    _item_cls,_label_cls = CollabLine,FloatList
    
    def reconstruct(self, t:Tensor): return CollabLine(t, [], self.classes, self.col_names)

class EmbeddingNN(TabularModel):
    "Subclass `TabularModel` to create a NN suitable for collaborative filtering."
    def __init__(self, emb_szs:ListSizes, **kwargs):
        super().__init__(emb_szs=emb_szs, n_cont=0, out_sz=1, **kwargs)

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        return super().forward(torch.stack([users,items], dim=1), None)

class EmbeddingDotBias(nn.Module):
    "Base dot model for collaborative filtering."
    def __init__(self, n_factors:int, n_users:int, n_items:int, y_range:Tuple[float,float]=None):
        super().__init__()
        self.y_range = y_range
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        dot = self.u_weight(users)* self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.y_range is None: return res
        return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]

class CollabDataBunch(DataBunch):
    @classmethod
    def from_df(cls, ratings:DataFrame, pct_val:float=0.2, user_name:Optional[str]=None, item_name:Optional[str]=None,
                rating_name:Optional[str]=None, test:DataFrame=None, seed=None, **kwargs):
        "Create a `DataBunch` suitable for collaborative filtering from `ratings`."
        user_name   = ifnone(user_name,  ratings.columns[0])
        item_name   = ifnone(item_name,  ratings.columns[1])
        rating_name = ifnone(rating_name,ratings.columns[2])
        cat_names = [user_name,item_name]
        src = (CollabList.from_df(ratings, cat_names=cat_names, procs=Categorify)
               .random_split_by_pct(valid_pct=pct_val, seed=seed).label_from_df(cols=rating_name))
        if test is not None: src.add_test(CollabList.from_df(test, cat_names=cat_names))
        return src.databunch(**kwargs)

class CollabLearner(Learner):
    def get_idx(self, arr:Collection, is_item:bool=True):
        "Fetch item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        m = self.model.eval().cpu()
        requires_grad(m,False)
        u_class,i_class = self.data.classes.values()
        classes = i_class if is_item else u_class
        c2i = {v:k for k,v in enumerate(classes)}
        return tensor([c2i[o] for o in arr])

    def bias(self, arr:Collection, is_item:bool=True):
        "Bias for item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        idx = self.get_idx(arr, is_item)
        m = self.model
        layer = m.i_bias if is_item else m.u_bias
        return layer(idx).squeeze()

    def weight(self, arr:Collection, is_item:bool=True):
        "Bias for item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        idx = self.get_idx(arr, is_item)
        m = self.model
        layer = m.i_weight if is_item else m.u_weight
        return layer(idx)

def collab_learner(data, n_factors:int=None, use_nn:bool=False, metrics=None,
                   emb_szs:Dict[str,int]=None, wd:float=0.01, **kwargs)->Learner:
    "Create a Learner for collaborative filtering on `data`."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    u,m = data.classes.values()
    if use_nn: model = EmbeddingNN(emb_szs=emb_szs, **kwargs)
    else:      model = EmbeddingDotBias(n_factors, len(u), len(m), **kwargs)
    return CollabLearner(data, model, metrics=metrics, wd=wd)

