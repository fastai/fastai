from .core import *

__all__ = ['URLs', 'Config', 'untar_data', 'download_data', 'datapath4file', 'url2name']

MODEL_URL = 'http://files.fast.ai/models/'
URL = 'http://files.fast.ai/data/examples/'
class URLs():
    "Global constants for dataset and model URLs."
    LOCAL_PATH = Path.cwd()
    S3 = 'https://s3.amazonaws.com/fast-ai-'
    S3_IMAGE = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_NLP = f'{S3}nlp/'
    S3_COCO = f'{S3}coco/'
    S3_MODEL = f'{S3}modelzoo/'
    COCO_SAMPLE = f'{S3_COCO}coco_sample'
    COCO_TINY = f'{URL}coco_tiny'
    MNIST_SAMPLE = f'{URL}mnist_sample'
    MNIST_TINY = f'{URL}mnist_tiny'
    IMDB = f'{S3_NLP}imdb'
    IMDB_SAMPLE = f'{URL}imdb_sample'
    HUMAN_NUMBERS = f'{URL}human_numbers'
    ADULT_SAMPLE = f'{URL}adult_sample'
    ML_SAMPLE = f'{URL}movie_lens_sample'
    PLANET_SAMPLE = f'{URL}planet_sample'
    BIWI_SAMPLE = f'{URL}biwi_sample'
    PLANET_TINY = f'{URL}planet_tiny'
    CIFAR = f'{URL}cifar10'
    WT103 = f'{S3_MODEL}wt103'
    WT103_1 = f'{S3_MODEL}wt103-1'
    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats'
    PETS = f'{S3_IMAGE}oxford-iiit-pet'
    MNIST = f'{S3_IMAGE}mnist_png'
    CAMVID = f'{S3_IMAGELOC}camvid'
    CAMVID_TINY = f'{URL}camvid_tiny'
    BIWI_HEAD_POSE = f"{S3_IMAGELOC}biwi_head_pose"
    LSUN_BEDROOMS = f'{S3_IMAGE}bedroom'

#TODO: This can probably be coded more shortly and nicely.
class Config():
    "Creates a default config file at `~/.fastai/config.yml`"
    DEFAULT_CONFIG_PATH = '~/.fastai/config.yml'
    DEFAULT_CONFIG = {
        'data_path': '~/.fastai/data',
        'model_path': '~/.fastai/models'
    }

    @classmethod
    def get_key(cls, key):
        "Get the path to `key` in the config file."
        return cls.get().get(key, cls.DEFAULT_CONFIG.get(key,None))

    @classmethod
    def get_path(cls, path):
        "Get the `path` in the config file."
        return _expand_path(cls.get_key(path))

    @classmethod
    def data_path(cls):
        "Get the path to data in the config file."
        return cls.get_path('data_path')

    @classmethod
    def model_path(cls):
        "Get the path to fastai pretrained models in the config file."
        return cls.get_path('model_path')

    @classmethod
    def get(cls, fpath=None, create_missing=True):
        "Retrieve the `Config` in `fpath`."
        fpath = _expand_path(fpath or cls.DEFAULT_CONFIG_PATH)
        if not fpath.exists() and create_missing: cls.create(fpath)
        assert fpath.exists(), f'Could not find config at: {fpath}. Please create'
        with open(fpath, 'r') as yaml_file: return yaml.load(yaml_file)

    @classmethod
    def create(cls, fpath):
        "Creates a `Config` from `fpath`."
        fpath = _expand_path(fpath)
        assert(fpath.suffix == '.yml')
        if fpath.exists(): return
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w') as yaml_file:
            yaml.dump(cls.DEFAULT_CONFIG, yaml_file, default_flow_style=False)

def _expand_path(fpath): return Path(fpath).expanduser()
def url2name(url): return url.split('/')[-1]
def _url2path(url, data=True):
    name = url2name(url)
    return datapath4file(name) if data else modelpath4file(name)
def _url2tgz(url, data=True):
    return datapath4file(f'{url2name(url)}.tgz') if data else modelpath4file(f'{url2name(url)}.tgz')

def modelpath4file(filename):
    "Return model path to `filename`, checking locally first then in the config file."
    local_path = URLs.LOCAL_PATH/'models'/filename
    if local_path.exists() or local_path.with_suffix('.tgz').exists(): return local_path
    else: return Config.model_path()/filename

def datapath4file(filename):
    "Return data path to `filename`, checking locally first then in the config file."
    local_path = URLs.LOCAL_PATH/'data'/filename
    if local_path.exists() or local_path.with_suffix('.tgz').exists(): return local_path
    else: return Config.data_path()/filename

def download_data(url:str, fname:PathOrStr=None, data:bool=True) -> Path:
    "Download `url` to destination `fname`."
    fname = Path(ifnone(fname, _url2tgz(url, data)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}.tgz', fname)
    return fname

def untar_data(url:str, fname:PathOrStr=None, dest:PathOrStr=None, data=True) -> Path:
    "Download `url` to `fname` if it doesn't exist, and un-tgz to folder `dest`."
    dest = Path(ifnone(dest, _url2path(url, data)))
    if not dest.exists():
        fname = download_data(url, fname=fname, data=data)
        tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest
