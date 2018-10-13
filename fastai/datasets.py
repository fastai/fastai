import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['Paths', 'untar_data', 'download_data']

URL = 'http://files.fast.ai/data/examples/'
class Paths():
    DATA = Path(__file__).parent/'..'/'data'
    MNIST = DATA/'mnist_sample'
    MNIST_TINY = DATA/'mnist_tiny'
    IMDB = DATA/'imdb_sample'
    ADULT = DATA/'adult_sample'
    ML = DATA/'movie_lens_sample'
    CIFAR = DATA/'cifar10'
    PLANET = DATA/'planet_sample'
    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = DATA/'dogscats'

def f_name(name): return f'{name}.tgz'

def download_data(name):
    os.makedirs(Paths.DATA, exist_ok=True)
    dest = Paths.DATA/f_name(name)
    if not dest.exists(): download_url(f'{URL}{f_name(name)}', dest)

def untar_data(path):
    download_data(path.name)
    if not path.exists(): tarfile.open(f_name(path), 'r:gz').extractall(Paths.DATA)

def get_adult():
    untar_data(Paths.ADULT)
    return pd.read_csv(Paths.ADULT/'adult.csv')

def get_mnist():
    untar_data(Paths.MNIST)
    return image_data_from_folder(Paths.MNIST)

def get_imdb(classifier=False):
    untar_data(Paths.IMDB)
    data_func = classifier_data if classifier else lm_data
    return text_data_from_csv(Paths.IMDB, data_func=data_func)

def get_movie_lens():
    untar_data(Paths.ML)
    return pd.read_csv(Paths.ML/'ratings.csv')

def download_wt103_model():
    model_path = Paths.IMDB/'models'
    os.makedirs(model_path, exist_ok=True)
    download_url('http://files.fast.ai/models/wt103_v1/lstm_wt103.pth', model_path/'lstm_wt103.pth')
    download_url('http://files.fast.ai/models/wt103_v1/itos_wt103.pkl', model_path/'itos_wt103.pkl')

