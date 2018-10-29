from ..torch_core import *
from ..layers import *

__all__ = ['TabularModel']

class TabularModel(nn.Module):
    "Basic model for tabular data."

    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int],
            ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        self.embeds = nn.ModuleList([get_embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * F.sigmoid(x) + self.y_range[0]
        return x.squeeze()

