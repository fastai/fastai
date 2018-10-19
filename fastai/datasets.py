import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['URLs', 'untar_data', 'download_data']

URL = 'http://files.fast.ai/data/examples/'
class URLs():
    S3 = 'https://s3.amazonaws.com/fast-ai-'
    S3_IMAGE = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_NLP = f'{S3}nlp/'
    S3_COCO = f'{S3}coco/'
    DATA = Path(__file__).parent/'..'/'data'
    MNIST_SAMPLE = f'{URL}mnist_sample'
    MNIST_TINY = f'{URL}mnist_tiny'
    IMDB_SAMPLE = f'{URL}imdb_sample'
    ADULT_SAMPLE = f'{URL}adult_sample'
    ML_SAMPLE = f'{URL}movie_lens_sample'
    PLANET_SAMPLE = f'{URL}planet_sample'
    CIFAR = f'{URL}cifar10'
    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats'
    PETS = f'{S3_IMAGE}oxford-iiit-pet'
    MNIST = f'{S3_IMAGE}mnist_png'

    @classmethod
    def get_adult(cls):
        path = untar_data(cls.ADULT_SAMPLE)
        return pd.read_csv(path/'adult.csv')

    @classmethod
    def get_mnist(cls):
        path = untar_data(cls.MNIST_SAMPLE)
        return ImageDataBunch.from_folder(path)

    @classmethod
    def get_imdb(cls, classifier=False):
        path = untar_data(cls.IMDB_SAMPLE)
        data_class = TextClasDataBunch if classifier else TextLMDataBunch
        return data_class.from_csv(path)

    @classmethod
    def get_movie_lens(cls):
        path = untar_data(cls.ML_SAMPLE)
        return pd.read_csv(path/'ratings.csv')

    @classmethod
    def download_wt103_model(cls):
        path = untar_data(cls.IMDB_SAMPLE)
        model_path = path/'models'
        model_path.mkdir(exist_ok=True)
        url = 'http://files.fast.ai/models/wt103_v1/'
        download_url(f'{url}lstm_wt103.pth', model_path/'lstm_wt103.pth')
        download_url(f'{url}itos_wt103.pkl', model_path/'itos_wt103.pkl')


def _url2name(url): return url.split('/')[-1]
def _url2path(url): return URLs.DATA/f'{_url2name(url)}'
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
    dest = Path(ifnone(dest, _url2path(url)))
    if not dest.exists(): tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest

