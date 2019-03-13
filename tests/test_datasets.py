import pytest, fastai, shutil
from fastai.gen_doc.doctest import this_tests
from fastai.datasets import *
from fastai.datasets import Config, _expand_path
from pathlib import Path


def clean_path(path):
    path = Path(path)
    if path.is_file(): path.unlink()
    if path.is_dir(): shutil.rmtree(path)

def test_creates_config():
    this_tests(Config)
    DEFAULT_CONFIG_PATH = 'config_test/test.yml'

    try:
        config_path = _expand_path(DEFAULT_CONFIG_PATH)
        clean_path(config_path)
        assert not config_path.exists(), "config path should not exist"
        config = Config.get(config_path)
        assert config_path.exists(), "Config.get should create config if it doesn't exist"
    finally:
        clean_path(config_path)
