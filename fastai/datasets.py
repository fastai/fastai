from .core import *

__all__ = ['URLs', 'untar_data', 'download_data', 'datapath4file']

MODEL_URL = 'http://files.fast.ai/models/'
URL = 'http://files.fast.ai/data/examples/'
class URLs():
    LOCAL_PATH = Path.cwd()
    S3 = 'https://s3.amazonaws.com/fast-ai-'
    S3_IMAGE = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_NLP = f'{S3}nlp/'
    S3_COCO = f'{S3}coco/'
    S3_MODEL = f'{S3}modelzoo/'
    MNIST_SAMPLE = f'{URL}mnist_sample'
    MNIST_TINY = f'{URL}mnist_tiny'
    IMDB_SAMPLE = f'{URL}imdb_sample'
    HUMAN_NUMBERS = f'{URL}human_numbers'
    ADULT_SAMPLE = f'{URL}adult_sample'
    ML_SAMPLE = f'{URL}movie_lens_sample'
    PLANET_SAMPLE = f'{URL}planet_sample'
    CIFAR = f'{URL}cifar10'
    WT103 = f'{S3_MODEL}wt103'
    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats'
    PETS = f'{S3_IMAGE}oxford-iiit-pet'
    MNIST = f'{S3_IMAGE}mnist_png'

class Config():
    "Creates a default config file at `~/.fastai/config.yml`"
    DEFAULT_CONFIG_PATH = '~/.fastai/config.yml'
    DEFAULT_CONFIG = {
        'data_path': '~/.fastai/data',
        'model_path': '~/.fastai/models'
    }

    @classmethod
    def get_key(cls, key): return cls.get().get(key, cls.DEFAULT_CONFIG.get(key,None))

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
def _url2path(url, data=True): return datapath4file(f'{_url2name(url)}') if data else modelpath4file(f'{_url2name(url)}')
def _url2tgz(url): return datapath4file(f'{_url2name(url)}.tgz')

def modelpath4file(filename):
    "Returns URLs.MODEL path if file exists. Otherwise returns config path"
    local_path = URLs.LOCAL_PATH/'models'/filename
    if local_path.exists() or local_path.with_suffix('.tgz').exists(): return local_path
    else: return _expand_path(Config.get_key('model_path'))/filename

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

def untar_data(url:str, fname:PathOrStr=None, dest:PathOrStr=None, data=True):
    "Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`"
    dest = Path(ifnone(dest, _url2path(url)))
    if not dest.exists():
        fname = download_data(url, fname=fname)
        tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest
