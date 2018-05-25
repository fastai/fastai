from pathlib import Path
import pytest

from PIL import Image
import pandas as pd
import numpy as np
import os

from fastai.dataset import ImageClassifierData
from fastai.model import resnet34
from fastai.transforms import tfms_from_model


@pytest.fixture(scope='module')
def root_folder(tmpdir_factory):
    return tmpdir_factory.mktemp('tmp_img_data')


@pytest.fixture(scope='module')
def csv_file(root_folder):
    tmp_csv_file = root_folder.mkdir(
        'tmp_csv_folder').join('tmp_csv_file.csv')
    df_list = [{'id': 11-i, 'label': chr(ord('a') + 10 - i)}
               for i in range(1, 11)]  # Create CSV with rows as (10, 'j'), (9, 'i') .. (1, 'a')
    df = pd.DataFrame(df_list)
    df.to_csv(str(tmp_csv_file), index=False)
    return tmp_csv_file


@pytest.fixture(scope='module')
def data_folder(root_folder):
    folder = root_folder.mkdir('tmp_data_folder')
    for i in range(1, 11):  # Create folder with images "1.png", "2.png".."10.png"
        img_array = np.random.rand(100, 100, 3) * 255
        img = Image.fromarray(img_array.astype('uint8')).convert('RGBA')
        img.save(str(folder.join(str(i) + '.png')))
    return folder


def test_image_classifier_data_from_csv_unsorted(root_folder, csv_file, data_folder):
    val_idxs = [2, 3]
    tfms = tfms_from_model(resnet34, 224)
    path = str(root_folder)
    folder = 'tmp_data_folder'
    csv_fname = Path(str(csv_file))
    data = ImageClassifierData.from_csv(path=Path(
        path), folder=folder, csv_fname=csv_fname, val_idxs=val_idxs, suffix='.png', tfms=tfms)
    val_fnames = ['8.png', '7.png']
    assert [os.path.split(o)[-1]
            for o in data.val_ds.fnames.tolist()] == val_fnames
