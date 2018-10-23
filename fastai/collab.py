"Module support for Collaborative Filtering"
from .torch_core import *
from .basic_train import *
from .basic_data import *
from .layers import *

__all__ = ['CollabFilteringDataset', 'EmbeddingDotBias', 'get_collab_learner']

@dataclass
class CollabFilteringDataset(DatasetBase):
    "Base dataset for collaborative filtering."
    user:Series
    item:Series
    ratings:np.ndarray
    def __post_init__(self):
        self.user_ids = np.array(self.user.cat.codes, dtype=np.int64)
        self.item_ids = np.array(self.item.cat.codes, dtype=np.int64)
        self.loss_func = F.mse_loss

    def __len__(self)->int: return len(self.ratings)

    def __getitem__(self, idx:int)->Tuple[Tuple[int,int],float]:
        return (self.user_ids[idx],self.item_ids[idx]), self.ratings[idx]

    @property
    def c(self) -> int: return 1

    @property
    def n_user(self)->int: return len(self.user.cat.categories)

    @property
    def n_item(self)->int: return len(self.item.cat.categories)

    @classmethod
    def from_df(cls, rating_df:DataFrame, pct_val:float=0.2, user_name:Optional[str]=None, item_name:Optional[str]=None,
                rating_name:Optional[str]=None) -> Tuple['ColabFilteringDataset','ColabFilteringDataset']:
        "Split a given dataframe in a training and validation set."
        if user_name is None:   user_name = rating_df.columns[0]
        if item_name is None:   item_name = rating_df.columns[1]
        if rating_name is None: rating_name = rating_df.columns[2]
        user = rating_df[user_name]
        item = rating_df[item_name]
        ratings = np.array(rating_df[rating_name], dtype=np.float32)
        idx = np.random.permutation(len(ratings))
        if pct_val is None: return cls(user, item, ratings)
        cut = int(pct_val * len(ratings))
        return (cls(user[idx[cut:]], item[idx[cut:]], ratings[idx[cut:]]),
                cls(user[idx[:cut]], item[idx[:cut]], ratings[idx[:cut]]))

    @classmethod
    def from_csv(cls, csv_name:str, **kwargs) -> Tuple['ColabFilteringDataset','ColabFilteringDataset']:
        "Split a given table in a csv in a training and validation set."
        df = pd.read_csv(csv_name)
        return cls.from_df(df, **kwargs)

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
          item_name:Optional[str]=None, rating_name:Optional[str]=None, test:DataFrame=None, metrics=None,
          min_score:float=None, max_score:float=None, **kwargs) -> Learner:
    "Create a Learner for collaborative filtering."
    datasets = list(CollabFilteringDataset.from_df(ratings, pct_val, user_name, item_name, rating_name))
    if test is not None:
        datasets.append(CollabFilteringDataset.from_df(test, None, user_name, item_name, rating_name))
    data = DataBunch.create(*datasets, **kwargs)
    model = EmbeddingDotBias(n_factors, datasets[0].n_user, datasets[0].n_item, min_score, max_score)
    return Learner(data, model, metrics=metrics)
