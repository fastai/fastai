"Module support for Collaborative Filtering"
from .torch_core import *
from .basic_train import *
from .basic_data import *
from .layers import *
from .tabular.data import *
from .tabular.transform import *

__all__ = ['EmbeddingDotBias', 'get_collab_learner', 'CollabLine', 'CollabList']

class CollabLine(TabularLine):
    def __init__(self, cats, conts, classes, names):
        super().__init__(cats, conts, classes, names)
        self.data = [self.data[0][0]-1,self.data[0][1]-1]

class CollabList(TabularList):
    _item_cls = CollabLine

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

def get_collab_learner(ratings:DataFrame, n_factors:int, pct_val:float=0.2, user_name:Optional[str]=None,
          item_name:Optional[str]=None, rating_name:Optional[str]=None, test:DataFrame=None,
          metrics=None, min_score:float=None, max_score:float=None, seed=None, **kwargs) -> Learner:
    "Create a Learner for collaborative filtering."
    user_name = ifnone(user_name,ratings.columns[0])
    item_name = ifnone(item_name,ratings.columns[1])
    rating_name = ifnone(rating_name,ratings.columns[2])
    src = (CollabList.from_df(ratings, cat_names=[user_name, item_name], procs=Categorify)
                     .random_split_by_pct(seed=seed)
                     .label_from_df(cols=rating_name))
    if test is not None: src.add_test(CollabList.from_df(test, cat_names=[user_name, item_name], cont_names=[]))
    data = src.databunch()
    model = EmbeddingDotBias(n_factors, len(data.classes[user_name]), len(data.classes[item_name]), min_score, max_score)
    return Learner(data, model, metrics=metrics)

