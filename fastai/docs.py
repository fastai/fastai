import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['DATA_PATH', 'MNIST_PATH', 'IMDB_PATH', 'ADULT_PATH', 'ML_PATH', 'DOGS_PATH', 'PLANET_PATH',
           'CIFAR_PATH', 'untar_data', 'get_adult', 'get_mnist', 'get_imdb', 'get_movie_lens', 'download_wt103_model']

URL = 'http://files.fast.ai/data/examples/'
DATA_PATH = Path('..')/'data'
MNIST_PATH = DATA_PATH/'mnist_sample'
IMDB_PATH = DATA_PATH/'imdb_sample'
ADULT_PATH = DATA_PATH/'adult_sample'
ML_PATH = DATA_PATH/'movie_lens_sample'
CIFAR_PATH = DATA_PATH/'cifar10'
PLANET_PATH = DATA_PATH/'planet_sample'
# kaggle competitions download dogs-vs-cats -p {DOGS_PATH.absolute()}
DOGS_PATH = DATA_PATH/'dogscats'

def f_name(name): return f'{name}.tgz'

def download_data(name):
    os.makedirs(DATA_PATH, exist_ok=True)
    dest = DATA_PATH/f_name(name)
    if not dest.exists(): download_url(f'{URL}{f_name(name)}', dest)

def untar_data(path):
    download_data(path.name)
    if not path.exists(): tarfile.open(f_name(path), 'r:gz').extractall(DATA_PATH)

def get_adult():
    untar_data(ADULT_PATH)
    return pd.read_csv(ADULT_PATH/'adult.csv')

def get_mnist():
    untar_data(MNIST_PATH)
    return image_data_from_folder(MNIST_PATH)

def get_imdb(classifier=False):
    untar_data(IMDB_PATH)
    data_func = classifier_data if classifier else lm_data
    return text_data_from_csv(IMDB_PATH, data_func=data_func)

def get_movie_lens():
    untar_data(ML_PATH)
    return pd.read_csv(ML_PATH/'ratings.csv')

def download_wt103_model():
    model_path = IMDB_PATH/'models'
    os.makedirs(model_path, exist_ok=True)
    download_url('http://files.fast.ai/models/wt103_v1/lstm_wt103.pth', model_path/'lstm_wt103.pth')
    download_url('http://files.fast.ai/models/wt103_v1/itos_wt103.pkl', model_path/'itos_wt103.pkl')

