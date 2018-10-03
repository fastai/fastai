from fast_gen import *
from learner import *
from pt_models import *
from dataset_pt import *
from sgdr_pt import *
from planet import *

bs=64; f_model = resnet34
path = "/data/jhoward/fast/planet/"
cv_idx = int(sys.argv[1])
torch.cuda.set_device(cv_idx % 4)
if cv_idx==1: torch.cuda.set_device(2)
n=len(list(open(f'{path}train_v2.csv')))-1

def train_sz(sz, load=None, save_name=None, suf=None):
    print(f'\n***** {sz} *****')
    #data=get_data_pad(f_model, path, sz, bs, n, cv_idx)
    data=get_data_zoom(f_model, path, sz, bs, n, cv_idx)
    learn = Learner.pretrained_convnet(f_model, data, metrics=[f2])
    if load: learn.load(f'{load}_{cv_idx}{suf}')
    print('--- FC')
    learn.fit(0.3, 2, cycle_len=1)
    print('--- Gradual')
    for i in range(6,3,-1):
        learn.freeze_to(i)
        learn.fit(0.1*(i-3), 1, cycle_len=1)
    learn.unfreeze()
    print('--- All')
    learn.fit(0.2, 15, cycle_len=3, cycle_save_name=f'{save_name}{suf}')
    learn.save(f'{sz}_{cv_idx}{suf}')

suf='_zoom'
train_sz(64, suf=suf)
train_sz(128, load=64, suf=suf)
train_sz(244, load=128, save_name=f'170809_{cv_idx}', suf=suf)

