from imports import *
from torch_imports import *
from dataset_pt import *
from learner import *

class ColumnarDataset(Dataset):
    def __init__(self,*args):
        *xs,y=args
        self.xs = [T(x) for x in xs]
        self.y = T(y).unsqueeze(1)
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return [o[idx] for o in self.xs] + [self.y[idx]]
    
    @classmethod
    def from_data_frame(self, df, cols_x, col_y):
        cols = [df[o] for o in cols_x+[col_y]]
        return self(*cols)

class ColumnarModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs): 
        super().__init__(path, DataLoader(trn_ds, bs, shuffle=True),
            DataLoader(val_ds, bs*2, shuffle=False))

    @classmethod
    def from_data_frames(self, path, trn_df, val_df, cols_x, col_y, bs):
        return self(path, ColumnarDataset.from_data_frame(trn_df, cols_x, col_y),
                    ColumnarDataset.from_data_frame(val_df, cols_x, col_y), bs)
    
    @classmethod
    def from_data_frame(self, path, val_idxs, df, cols_x, col_y, bs):
        ((val_df, trn_df),) = split_by_idx(val_idxs, df)
        return self.from_data_frames(path, trn_df, val_df, cols_x, col_y, bs)


class CollabFilterDataset(Dataset):
    def __init__(self, path, user_col, item_col, ratings):
        self.ratings,self.path = ratings,path
        self.n = len(ratings)
        (self.users,self.user2idx,self.user_col,self.n_users) = self.proc_col(user_col)
        (self.items,self.item2idx,self.item_col,self.n_items) = self.proc_col(item_col)
        self.min_score,self.max_score = min(ratings),max(ratings)
        self.cols = [self.user_col,self.item_col,self.ratings]

    @classmethod
    def from_data_frame(self, path, df, user_name, item_name, rating_name):
        return self(path, df[user_name], df[item_name], df[rating_name])

    @classmethod
    def from_csv(self, path, csv, user_name, item_name, rating_name):
        df = pd.read_csv(os.path.join(path,csv))
        return self.from_data_frame(path, df, user_name, item_name, rating_name)

    def proc_col(self,col):
        uniq = col.unique()
        name2idx = {o:i for i,o in enumerate(uniq)}
        return (uniq, name2idx, np.array([name2idx[x] for x in col]), len(uniq))
        
    def __len__(self): return self.n
    def __getitem__(self, idx): return [o[idx] for o in self.cols]

    def get_data(self, val_idxs, bs):
        val, trn = zip(*split_by_idx(val_idxs, *self.cols))
        return ColumnarModelData(self.path, ColumnarDataset(*trn), ColumnarDataset(*val), bs)

    def get_model(self, n_factors):
        model = EmbeddingDotBias(n_factors, self.n_users, self.n_items, self.min_score, self.max_score)
        return CollabFilterModel(model.cuda())

    def get_learner(self, n_factors, val_idxs, bs, **kwargs):
        return CollabFilterLearner(self.get_data(val_idxs, bs), self.get_model(n_factors), **kwargs)


def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.05,0.05)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self, n_factors, n_users, n_items, min_score, max_score):
        super().__init__()
        self.min_score,self.max_score = min_score,max_score
        (self.u, self.i, self.ub, self.ib) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]
        
    def forward(self, users, items):
        um = self.u(users)* self.i(items)
        res = um.sum(1) + self.ub(users).squeeze() + self.ib(items).squeeze()
        return F.sigmoid(res) * (self.max_score-self.min_score) + self.min_score

class CollabFilterLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.mse_loss

class CollabFilterModel():
    def __init__(self,model):
        self.model=model

    def get_layer_groups(self):
        return list(split_by_idxs(self.children,[2]))
