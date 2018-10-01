import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['DATA_PATH', 'MNIST_NAME', 'IMDB_NAME', 'ADULT_NAME', 'ML_NAME',
           'MNIST_PATH', 'IMDB_PATH', 'ADULT_PATH', 'ML_PATH',
           'untar_mnist', 'untar_imdb', 'untar_adult', 'untar_movie_lens',
           'get_adult', 'get_mnist', 'get_imdb', 'get_movie_lens', 'download_wt103_model']

URL = 'http://files.fast.ai/data/examples/'
DATA_PATH = Path('..')/'data'
MNIST_NAME = 'mnist_sample'
IMDB_NAME = 'imdb_sample'
ADULT_NAME = 'adult_sample'
ML_NAME = 'movie_lens_sample'
MNIST_PATH = DATA_PATH/MNIST_NAME
IMDB_PATH = DATA_PATH/IMDB_NAME
ADULT_PATH = DATA_PATH/ADULT_NAME
ML_PATH = DATA_PATH/ML_NAME

def f_name(name): return f'{name}.tgz'

def download_data(name):
    dest = DATA_PATH/f_name(name)
    if not dest.exists(): download_url(f'{URL}{f_name(name)}', dest)

def untar_data(name):
    download_data(name)
    if not (DATA_PATH/name).exists(): tarfile.open(f_name(DATA_PATH/name), 'r:gz').extractall(DATA_PATH)

def untar_mnist(): untar_data(MNIST_NAME)
def untar_imdb(): untar_data(IMDB_NAME)
def untar_adult(): untar_data(ADULT_NAME)
def untar_movie_lens(): untar_data(ML_NAME)

def get_adult():
    untar_adult()
    return pd.read_csv(ADULT_PATH/'adult.csv')

def get_mnist():
    untar_mnist()
    return image_data_from_folder(MNIST_PATH)

def get_imdb(classifier=False):
    untar_imdb()
    data_func = classifier_data if classifier else lm_data
    return text_data_from_csv(IMDB_PATH, data_func=data_func)

def get_movie_lens():
    untar_movie_lens()
    return pd.read_csv(ML_PATH/'ratings.csv')

def download_wt103_model():
    model_path = IMDB_PATH/'models'
    os.makedirs(model_path, exist_ok=True)
    download_url('http://files.fast.ai/models/wt103_v1/lstm_wt103.pth', model_path/'lstm_wt103.pth')
    download_url('http://files.fast.ai/models/wt103_v1/itos_wt103.pkl', model_path/'itos_wt103.pkl')

