import pytest, fastai, shutil, os, yaml, sys
from fastai.gen_doc.doctest import this_tests
from fastai.datasets import *
from fastai.datasets import Config, _expand_path
from pathlib import Path


def clean_test_config(path):
    path = Path(path)
    if path.is_file(): shutil.rmtree(path.parent)
    if path.is_dir(): print(path); shutil.rmtree(path)
    Config.DEFAULT_CONFIG_LOCATION = os.path.expanduser(os.getenv('FASTAI_HOME', '~/.fastai'))
    Config.DEFAULT_CONFIG_PATH = Config.DEFAULT_CONFIG_LOCATION + '/config.yml'

def test_creates_config():
    this_tests(Config)
    DEFAULT_CONFIG_PATH = 'config_test/test.yml'

    try:
        config_path = _expand_path(DEFAULT_CONFIG_PATH)
        clean_test_config(config_path)
        assert not config_path.exists(), "config path should not exist"
        config = Config.get(config_path)
        assert config_path.exists(), "Config.get should create config if it doesn't exist"
    finally:
        clean_test_config(config_path)
        assert not config_path.exists(), "config path should not exist"

def test_default_config():
    this_tests(Config)
    Config.DEFAULT_CONFIG_LOCATION = 'config_test'
    Config.DEFAULT_CONFIG_PATH = Config.DEFAULT_CONFIG_LOCATION + '/config.yml'
    try:
        assert Config.get() == {
            'data_archive_path': str(_expand_path('~/.fastai/data')),
            'data_path': str(_expand_path('~/.fastai/data')),
            'model_path': str(_expand_path('~/.fastai/models'))
        }
    finally:
        clean_test_config(Config.DEFAULT_CONFIG_LOCATION)

@pytest.mark.slow
def test_user_config():
    this_tests(Config, download_data, untar_data, url2path, datapath4file)
    Config.DEFAULT_CONFIG_LOCATION = 'config_test'
    Config.DEFAULT_CONFIG_PATH = Config.DEFAULT_CONFIG_LOCATION + '/config.yml'
    clean_test_config(Config.DEFAULT_CONFIG_LOCATION)

    USER_CONFIG = {
            'data_archive_path': Config.DEFAULT_CONFIG_LOCATION + '/archive',
            'data_path': Config.DEFAULT_CONFIG_LOCATION + '/data_test',
            'model_path': Config.DEFAULT_CONFIG_LOCATION + '/model_test'
    }
    os.makedirs(Config.DEFAULT_CONFIG_LOCATION, exist_ok=True)
    with open(Config.DEFAULT_CONFIG_PATH, 'w+') as config_file:
        yaml.dump(USER_CONFIG, config_file)
    
    try:
        # No directory should be created before data download
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/data_test').exists()
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/archive').exists()
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/model_test').exists()
        download_data(URLs.MNIST_TINY)
        # Data directory should not be created in download_data
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/data_test').exists()
        # Compressed file should be saved to corresponding directories
        assert Path(Config.DEFAULT_CONFIG_LOCATION + '/archive/mnist_tiny.tgz').exists()
        untar_data(URLs.MNIST_TINY)
        # Data should be decompressed to corresponding directories
        assert Path(Config.DEFAULT_CONFIG_LOCATION + '/data_test/mnist_tiny').exists()
        # untar_data should not meddle with archive file if it exists and isn't corrupted
        assert Path(Config.DEFAULT_CONFIG_LOCATION + '/archive/mnist_tiny.tgz').exists()

        # No file should exist prior to download
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/data_test/mnist_sample').exists()
        assert not Path(Config.DEFAULT_CONFIG_LOCATION + '/archive/mnist_sample.tgz').exists()
        untar_data(URLs.ML_SAMPLE)
        # untar_data on dataset without local archive downloads the data too
        assert Path(Config.DEFAULT_CONFIG_LOCATION + '/data_test/movie_lens_sample').exists()
        assert Path(Config.DEFAULT_CONFIG_LOCATION + '/archive/movie_lens_sample.tgz').exists()
    finally:
        clean_test_config(Config.DEFAULT_CONFIG_LOCATION)