import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['URLs', 'untar_data', 'download_data', 'url2path', 'download_wt103_model',
           'get_mnist', 'get_imdb', 'get_adult', 'get_movie_lens']

URL = 'http://files.fast.ai/data/examples/'
S3_URL = 'https://s3.amazonaws.com/fast-ai-'
S3_IMAGE_URL = f'{S3_URL}imageclas/'
class URLs():
    DATA = Path(__file__).parent/'..'/'data'
    MNIST_SAMPLE = f'{URL}mnist_sample'
    MNIST_TINY = f'{URL}mnist_tiny'
    IMDB = f'{URL}imdb_sample'
    ADULT = f'{URL}adult_sample'
    ML = f'{URL}movie_lens_sample'
    CIFAR = f'{URL}cifar10'
    PLANET = f'{URL}planet_sample'
    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats'
    PETS = f'{S3_IMAGE_URL}oxford-iiit-pet'
    MNIST = f'{S3_IMAGE_URL}mnist_png'

def _url2name(url): return url.split('/')[-1]
def url2path(url): return URLs.DATA/f'{_url2name(url)}'
def _url2tgz(url): return URLs.DATA/f'{_url2name(url)}.tgz'

def download_data(url:str, fname:PathOrStr=None):
    "Download `url` to destination `fname`"
    fname = Path(ifnone(fname, _url2tgz(url)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}.tgz', fname)
    return fname

def untar_data(url:str, fname:PathOrStr=None, dest:PathOrStr=None):
    "Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`"
    fname = download_data(url, fname=fname)
    dest = Path(ifnone(dest, url2path(url)))
    if not dest.exists(): tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest

def get_adult():
    path = untar_data(URLs.ADULT)
    return pd.read_csv(path/'adult.csv')

def get_mnist():
    path = untar_data(URLs.MNIST_SAMPLE)
    return ImageDataBunch.from_folder(path)

def get_imdb(classifier=False):
    path = untar_data(URLs.IMDB)
    data_class = TextClasDataBunch if classifier else TextLMDataBunch
    return data_class.from_csv(path)

def get_movie_lens():
    path = untar_data(URLs.ML)
    return pd.read_csv(path/'ratings.csv')

def download_wt103_model():
    path = untar_data(URLs.IMDB)
    model_path = path/'models'
    model_path.mkdir(exist_ok=True)
    url = 'http://files.fast.ai/models/wt103_v1/'
    download_url(f'{url}lstm_wt103.pth', model_path/'lstm_wt103.pth')
    download_url(f'{url}itos_wt103.pkl', model_path/'itos_wt103.pkl')

