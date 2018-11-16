"Module support for Collaborative Filtering"
from .torch_core import *
from .basic_train import *
from .basic_data import *
from .layers import *
from .tabular import *

__all__ = ['EmbeddingDotBias', 'get_collab_learner', 'CollabDataBunch', 'CollabLine', 'CollabList']

class CollabLine(TabularLine):
    def __init__(self, cats, conts, classes, names):
        super().__init__(cats, conts, classes, names)
        self.data = [self.data[0][0]-1,self.data[0][1]-1]

class CollabList(TabularList): _item_cls = CollabLine

class EmbeddingDotBias(nn.Module):
    "Base model for callaborative filtering."
    def __init__(self, n_factors:int, n_users:int, n_items:int, min_score:float=None, max_score:float=None):
        super().__init__()
        self.min_score,self.max_score = min_score,max_score
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [get_embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        dot = self.u_weight(users)* self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.min_score is None: return res
        return torch.sigmoid(res) * (self.max_score-self.min_score) + self.min_score

class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs:ListSizes, **kwargs):
        super().__init__(emb_szs=emb_szs, n_cont=0, out_sz=1, **kwargs)

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        x_cat = torch.stack([users,items], dim=1)
        return super().forward(x_cat, None)


class CollabDataBunch(DataBunch):
    @classmethod
    def from_df(cls, ratings:DataFrame, pct_val:float=0.2, user_name:Optional[str]=None, item_name:Optional[str]=None,
                rating_name:Optional[str]=None, test:DataFrame=None, seed=None, **kwargs):
        user_name   = ifnone(user_name,  ratings.columns[0])
        item_name   = ifnone(item_name,  ratings.columns[1])
        rating_name = ifnone(rating_name,ratings.columns[2])
        cat_names = [user_name,item_name]
        src = (CollabList.from_df(ratings, cat_names=cat_names, procs=Categorify)
                 .random_split_by_pct(seed=seed).label_from_df(cols=rating_name))
        if test is not None: src.add_test(CollabList.from_df(test, cat_names=cat_names))
        return src.databunch(**kwargs)

def get_collab_learner(data, n_factors:int=None, use_nn:bool=False, metrics=None, min_score:float=None,
                       emb_szs:Dict[str,int]=None, max_score:float=None, **kwargs)->Learner:
    "Create a Learner for collaborative filtering."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    u,m = data.classes.values()
    if use_nn: model = EmbeddingNN(emb_szs=emb_szs, y_range=[min_score,max_score], **kwargs)
    else:      model = EmbeddingDotBias(n_factors, len(u), len(m), min_score, max_score)
    return Learner(data, model, metrics=metrics)

