from .core import *
import hashlib

__all__ = ['URLs', 'Config', 'untar_data', 'download_data', 'datapath4file', 'url2name', 'url2path']

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
    MNIST_VAR_SIZE_TINY = f'{S3_IMAGE}mnist_var_size_tiny'
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
    OPENAI_TRANSFORMER = f'{S3_MODEL}transformer'

# to create/update a checksum for ./mnist_var_size_tiny.tgz, run:
# python -c 'import fastai.datasets; print(fastai.datasets._check_file("mnist_var_size_tiny.tgz"))'
_checks = {URLs.COCO_SAMPLE:(3245877008, '006cd55d633d94b36ecaf661467830ec'),
           URLs.COCO_TINY:(801038, '367467451ac4fba79a647753c2c66d3a'),
           URLs.MNIST_SAMPLE:(3214948, '2dbc7ec6f9259b583af0072c55816a88'),
           URLs.MNIST_TINY:(342207, '56143e8f24db90d925d82a5a74141875'),
           URLs.MNIST_VAR_SIZE_TINY:(565372, 'b71a930f4eb744a4a143a6c7ff7ed67f'),
           URLs.IMDB:(144440600, '90f9b1c4ff43a90d67553c9240dc0249'),
           URLs.IMDB_SAMPLE:(571827, '0842e61a9867caa2e6fbdb14fa703d61'),
           URLs.HUMAN_NUMBERS:(30252, '8a19c3bfa2bcb08cd787e741261f3ea2'),
           URLs.ADULT_SAMPLE:(968212, '64eb9d7e23732de0b138f7372d15492f'),
           URLs.ML_SAMPLE:(51790, '10961384dfe7c5181460390a460c1f77'),
           URLs.PLANET_SAMPLE:(15523994, '8bfb174b3162f07fbde09b54555bdb00'),
           URLs.BIWI_SAMPLE:(593774, '9179f4c1435f4b291f0d5b072d60c2c9'),
           URLs.PLANET_TINY:(997569, '490873c5683454d4b2611fb1f00a68a9'),
           URLs.CIFAR:(168168549, 'a5f8c31371b63a406b23368042812d3c'),
           URLs.WT103:(206789489, '76fd08236c78bf91b7fb76698d53afa3'),
           URLs.WT103_1:(165175630, '9cbe02e9e23b969fee10dc9b8dec6566'),
           URLs.DOGS:(839285364, '3e483c8d6ef2175e9d395a6027eb92b7'),
           URLs.PETS:(811706944, 'e4db5c768afd933bb91f5f594d7417a4'),
           URLs.MNIST:(15683414, '03639f83c4e3d19e0a3a53a8a997c487'),
           URLs.CAMVID:(598913237, '648371e4f3a833682afb39b08a3ce2aa'),
           URLs.CAMVID_TINY:(2314212, '2cf6daf91b7a2083ecfa3e9968e9d915'),
           URLs.BIWI_HEAD_POSE:(452316199, '00f4ccf66e8cba184bc292fdc08fb237'),
           URLs.LSUN_BEDROOMS:(4579163978, '35d84f38f8a15fe47e66e460c8800d68'),
           URLs.OPENAI_TRANSFORMER:(432848315, '024b0d2203ebb0cd1fc64b27cf8af18e')}

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
def url2path(url, data=True):
    "Change `url` to a path."
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

def _check_file(fname):
    size = os.path.getsize(fname)
    with open(fname, "rb") as f:
        hash_nb = hashlib.md5(f.read(2**20)).hexdigest()
    return size,hash_nb

def untar_data(url:str, fname:PathOrStr=None, dest:PathOrStr=None, data=True, force_download=False) -> Path:
    "Download `url` to `fname` if it doesn't exist, and un-tgz to folder `dest`."
    dest = url2path(url, data) if dest is None else Path(dest)/url2name(url)
    fname = Path(ifnone(fname, _url2tgz(url, data)))
    if force_download or (fname.exists() and url in _checks and _check_file(fname) != _checks[url]):
        print(f"A new version of the {'dataset' if data else 'model'} is available.")
        os.remove(fname)
        shutil.rmtree(dest)
    if not dest.exists():
        fname = download_data(url, fname=fname, data=data)
        data_dir = Config().data_path()
        assert _check_file(fname) == _checks[url], f"Downloaded file {fname} does not match checksum expected! Remove that file from {data_dir} and try your code again."
        tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest
