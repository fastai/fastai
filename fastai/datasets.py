import tarfile
from fastai import *
from fastai.vision import *
from fastai.text import *

__all__ = ['URLs', 'untar_data', 'download_data', 'datapath4file']

URL = 'http://files.fast.ai/data/examples/'
class URLs():
    LOCAL_PATH = Path.cwd()
    S3 = 'https://s3.amazonaws.com/fast-ai-'
    S3_IMAGE = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_NLP = f'{S3}nlp/'
    S3_COCO = f'{S3}coco/'
    MNIST_SAMPLE = f'{URL}mnist_sample'
    MNIST_TINY = f'{URL}mnist_tiny'
    IMDB_SAMPLE = f'{URL}imdb_sample'
    HUMAN_NUMBERS = f'{URL}human_numbers'
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

class Config():
    "Creates a default config file at `~/.fastai/config.yml`"
    DEFAULT_CONFIG_PATH = '~/.fastai/config.yml'
    DEFAULT_CONFIG = {
        'data_path': '~/.fastai/data'
    }

    @classmethod
    def get_key(cls, key): return cls.get().get(key)

    @classmethod
    def get(cls, fpath=None, create_missing=True):
        fpath = _expand_path(fpath or cls.DEFAULT_CONFIG_PATH)
        if not fpath.exists() and create_missing: cls.create(fpath)
        assert fpath.exists(), f'Could not find config at: {fpath}. Please create'
        with open(fpath, 'r') as yaml_file:
            return yaml.load(yaml_file)

    @classmethod
    def create(cls, fpath):
        fpath = _expand_path(fpath)
        assert(fpath.suffix == '.yml')
        if fpath.exists(): return
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w') as yaml_file:
            yaml.dump(cls.DEFAULT_CONFIG, yaml_file, default_flow_style=False)

def _expand_path(fpath): return Path(fpath).expanduser()
def _url2name(url): return url.split('/')[-1]
def _url2path(url): return datapath4file(f'{_url2name(url)}')
def _url2tgz(url): return datapath4file(f'{_url2name(url)}.tgz')

def datapath4file(filename):
    "Returns URLs.DATA path if file exists. Otherwise returns config path"
    local_path = URLs.LOCAL_PATH/'data'/filename
    if local_path.exists() or local_path.with_suffix('.tgz').exists(): return local_path
    else: return _expand_path(Config.get_key('data_path'))/filename

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
    dest = Path(ifnone(dest, _url2path(url)))
    fname = download_data(url, fname=fname)
    if not dest.exists(): tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest
