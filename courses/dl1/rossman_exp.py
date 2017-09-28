train_ratio=0.9
use_dict=True
use_scaler=False
init_emb=False
split_contins=True
samp_size = 100000
#samp_size = 0

import math, keras, datetime, pandas as pd, numpy as np, keras.backend as K
import matplotlib.pyplot as plt, xgboost, operator, random, pickle, os
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from keras.models import Model
from keras.layers import merge, Input
from keras.layers.core import Dense, Activation, Reshape, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import initializations
np.set_printoptions(4)

cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

os.chdir('data/rossman')
cat_var_dict = {'Store': 50, 'DayOfWeek': 6, 'Year': 2, 'Month': 6,
    'Day': 10, 'StateHoliday': 3, 'CompetitionMonthsOpen': 2,
    'Promo2Weeks': 1, 'StoreType': 2, 'Assortment': 3, 'PromoInterval': 3,
    'CompetitionOpenSinceYear': 4, 'Promo2SinceYear': 4, 'State': 6,
    'Week': 2, 'Events': 4, 'Promo_fw': 1,
    'Promo_bw': 1, 'StateHoliday_fw': 1,
    'StateHoliday_bw': 1, 'SchoolHoliday_fw': 1,
    'SchoolHoliday_bw': 1}

cats, contins= [o for n,o in np.load('vars.npz').items()]
y = np.load('deps.npz').items()[0][1]

if samp_size != 0:
    np.random.seed(42)
    idxs = sorted(np.random.choice(len(y), samp_size, replace=False))
    cats= cats[idxs]
    contins= contins[idxs]
    y= y[idxs]

n=len(y)
train_size = int(n*train_ratio)

contins_trn_orig, contins_val_orig = contins[:train_size], contins[train_size:]
cats_trn, cats_val = cats[:train_size], cats[train_size:]
y_trn, y_val = y[:train_size], y[train_size:]

contin_map_fit = pickle.load(open('contin_maps.pickle', 'rb'))
cat_map_fit = pickle.load(open('cat_maps.pickle', 'rb'))

def cat_map_info(feat): return feat[0], len(feat[1].classes_)

co_enc = StandardScaler().fit(contins_trn_orig)
tf_contins_trn = co_enc.transform(contins_trn_orig)
tf_contins_val = co_enc.transform(contins_val_orig)


"""
def rmspe(y_pred, targ = y_valid_orig):
    return math.sqrt(np.square((targ - y_pred)/targ).mean())
def log_max_inv(preds, mx = max_log_y): return np.exp(preds * mx)
def normalize_inv(preds): return preds * ystd + ymean
"""


def split_cols(arr): return np.hsplit(arr,arr.shape[1])


def emb_init(shape, name=None):
    return initializations.uniform(shape, scale=0.6/shape[1], name=name)


def get_emb(feat):
    name, c = cat_map_info(feat)
    if use_dict:
        c2 = cat_var_dict[name]
    else:
        c2 = (c+2)//3
        if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=name+'_in')
    if init_emb:
        u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1)(inp))
    else:
        u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1, init=emb_init)(inp))
    return inp,u


def get_contin(feat):
    name = feat[0][0]
    inp = Input((1,), name=name+'_in')
    return inp, Dense(1, name=name+'_d')(inp)


def split_data():
    if split_contins:
        map_train = split_cols(cats_trn) + split_cols(contins_trn)
        map_valid = split_cols(cats_val) + split_cols(contins_val)
    else:
        map_train = split_cols(cats_trn) + [contins_trn]
        map_valid = split_cols(cats_val) + [contins_val]
    return (map_train, map_valid)


def get_contin_one():
    n_contin = contins_trn.shape[1]
    contin_inp = Input((n_contin,), name='contin')
    contin_out = BatchNormalization()(contin_inp)
    return contin_inp, contin_out


def train(model, map_train, map_valid,  bs=128, ne=10):
    return model.fit(map_train, y_trn, batch_size=bs, nb_epoch=ne,
                 verbose=0, validation_data=(map_valid, y_val))


def get_model():
    if split_contins:
        conts = [get_contin(feat) for feat in contin_map_fit.features]
        cont_out = [d for inp,d in conts]
        cont_inp = [inp for inp,d in conts]
    else:
        contin_inp, contin_out = get_contin_one()
        cont_out = [contin_out]
        cont_inp = [contin_inp]

    embs = [get_emb(feat) for feat in cat_map_fit.features]
    x = merge([emb for inp,emb in embs] + cont_out, mode='concat')

    x = Dropout(0.02)(x)
    x = Dense(1000, activation='relu', init='uniform')(x)
    x = Dense(500, activation='relu', init='uniform')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([inp for inp,emb in embs] + cont_inp, x)
    model.compile('adam', 'mean_absolute_error')
    #model.compile(Adam(), 'mse')
    return model

for split_contins in [True, False]:
    for use_dict in [True, False]:
        for use_scaler in [True, False]:
            for init_emb in [True, False]:
                print ({'split_contins':split_contins, 'use_dict':use_dict,
                       'use_scaler':use_scaler, 'init_emb':init_emb})
                if use_scaler:
                    contins_trn = tf_contins_trn
                    contins_val = tf_contins_val
                else:
                    contins_trn = contins_trn_orig
                    contins_val = contins_val_orig

                map_train, map_valid = split_data()
                model = get_model()
                hist = np.array(train(model, map_train, map_valid, 128, 10)
                                .history['val_loss'])
                print(hist)
                print(hist.min())

