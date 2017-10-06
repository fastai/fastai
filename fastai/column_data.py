from .imports import *
from .torch_imports import *
from .dataset import *
from .learner import *


class ColumnarDataset(Dataset):
    def __init__(self, cats, conts, y):
        self.cats = torch.stack([T(x) for x in cats], 1)
        self.conts = torch.stack([T(x) for x in conts], 1)
        self.y = T(y).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y):
        cat_cols = [c.values for n,c in df_cat.items()]
        cont_cols = [c.values for n,c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y)


class ColumnarModelData(ModelData):
    def __init__(self, path, trn_ds, val_ds, bs):
        super().__init__(path, DataLoader(trn_ds, bs, shuffle=True),
            DataLoader(val_ds, bs*2, shuffle=False))

    @classmethod
    def from_data_frames(self, path, trn_df, val_df, trn_y, val_y, cat_flds, bs):
        return self(path, ColumnarDataset.from_data_frame(trn_df, cat_flds, trn_y),
                    ColumnarDataset.from_data_frame(val_df, cat_flds, val_y), bs)

    @classmethod
    def from_data_frame(self, path, val_idxs, df, y, cat_flds, bs):
        ((val_df, trn_df), (val_y, trn_y)) = split_by_idx(val_idxs, df, y)
        return self.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, bs)

    def get_learner(self, emb_szs, n_cont, emb_drop, szs, drops):
        model = MixedInputModel(emb_szs, n_cont, emb_drop, szs, drops)
        return StructuredLearner(self, StructuredModel(model.cuda()), opt_fn=optim.Adam)


class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, szs, drops):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        n_emb = sum(e.embedding_dim for e in self.embs)
        szs = [n_emb+n_cont] + szs + [1]
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = [nn.Dropout(drop) for drop in drops]
        self.drops = nn.ModuleList(self.drops + [None])

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
        x = self.emb_drop(torch.cat(x, 1))
        x = torch.cat([x, x_cont], 1)
        for l,d in zip(self.lins, self.drops):
            x = F.relu(l(x))
            if d: x = d(x)
        return x


class StructuredLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.mse_loss

class StructuredModel(BasicModel):
    def get_layer_groups(self): return [self.model.embs, self.model.lins]


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

class CollabFilterModel(BasicModel):
    def get_layer_groups(self): return list(split_by_idxs(self.children,[2]))

