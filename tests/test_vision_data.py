import pytest
from fastai import *
from fastai.vision import *

def image_file(path):
    file = open(path, 'w')
    file.close()
    return file

def test_from_folder():
    n_classes = 2
    for valid_pct in [None, 0.5]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tmp')
        folder_paths = [os.path.join(path, folder) for folder in ['train', 'valid', 'test']]
        class_paths = [os.path.join(folder_paths[0], str(i)) for i in range(n_classes)]
        for each in [path]+folder_paths+class_paths: os.makedirs(each)
        for each in class_paths:
            for i in range(10): image_file(os.path.join(each, 'image%d.png'%i))
        try:
            data = ImageDataBunch.from_folder(path, test='test', valid_pct=valid_pct)
            assert len(data.classes) == n_classes
            assert set(data.classes) == set([str(i) for i in range(n_classes)])
        finally:
            shutil.rmtree(path)