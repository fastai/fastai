from learner import *


class ConvnetBuilder():
    """Class representing a convolutional network.    

    Arguments:
        f (string): name of the model  
        c (int): size of the last layer
        is_multi (bool): is multilabel classification
            (def here http://scikit-learn.org/stable/modules/multiclass.html)
        is_reg (bool): is a regression
        ps (float or array of float): dropout parameter
        xtra_fc (list of ints): list of hidden layers with hidden neurons
        xtra_cut (int): where to cut the model, detault is 0
    """
    model_meta = {
        resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnext50:[8,6],
        wrn:[8,6], dn121:[10,6], inceptionresnet_2:[5,9], inception_4:[19,8]
    }

    def __init__(self, f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0):
        self.f,self.c,self.is_multi,self.is_reg,self.xtra_cut = f,c,is_multi,is_reg,xtra_cut
        self.ps = ps or [0.25,0.5]
        self.xtra_fc = xtra_fc or [512]

        cut,self.lr_cut = self.model_meta[self.f]
        cut-=xtra_cut
        layers = cut_model(self.f(True), cut)
        self.nf=num_features(layers[-1])*2
        layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.ps, list): self.ps = [self.ps]*n_fc

        fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = nn.Sequential(*fc_layers).cuda()
        apply_init(self.fc_model, kaiming_normal)
        self.model=nn.Sequential(*(layers+fc_layers)).cuda()

    @property
    def name(self): return f'{self.f.__name__}_{self.xtra_cut}'

    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn())
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU)
            ni=nf
        final_actn = nn.Sigmoid if self.is_multi else nn.LogSoftmax
        if self.is_reg: final_actn = None 
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc: m,idxs = self.fc_model,[]
        else:     m,idxs = self.model,[self.lr_cut,-self.n_fc]
        lgs = list(split_by_idxs(children(m),idxs))
        return lgs


class ConvLearner(Learner):
    def __init__(self, data, models, use_fc=False, **kwargs):
        self.use_fc=False
        super().__init__(data, models, **kwargs)
        self.crit = F.binary_cross_entropy if data.is_multi else F.nll_loss
        if self.metrics is None and not data.is_reg:
            self.metrics = [accuracy_multi] if self.data.is_multi else [accuracy]
        if data.is_reg:
            self.crit = F.l1_loss 
        self.save_fc1()
        self.freeze()
        self.use_fc=use_fc

    @classmethod
    def pretrained(self, f, data, ps=None, xtra_fc=None, xtra_cut=0, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg, ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut)
        return self(data, models, **kwargs)

    @property
    def model(self): return self.models.fc_model if self.use_fc else self.models.model

    @property
    def data(self): return self.fc_data if self.use_fc else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data):
        super().set_data(data)
        self.save_fc1()
        self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.use_fc)

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val')]
        if os.path.exists(names[0]) and not force: 
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf,n) for n in names]

    def save_fc1(self):
        self.get_activations()
        act, val_act = self.activations

        if len(self.activations[0])==0:
            m=self.models.top_model
            predict_to_bcolz(m, self.data.fix_dl, act)
            predict_to_bcolz(m, self.data.val_dl, val_act)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
               (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs, classes=self.data.classes)

    def freeze(self): self.freeze_to(-self.models.n_fc)
