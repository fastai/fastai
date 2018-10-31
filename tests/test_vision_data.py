import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def prep_mnist_tiny_csv_labels(path):
    labels_df = pd.read_csv(path/'labels.csv')
    labels_df['label'] = labels_df['label'].apply(lambda val: '3' if val==0 else '7')
    return labels_df

def mnist_tiny_sanity_test(data):
    assert data.c == 2
    assert set(map(str, set(data.classes))) == {'3', '7'}
    assert set(data.train_ds.y) == set(data.valid_ds.y) == {0, 1}

def test_from_folder(path):
    for valid_pct in [None, 0.9]:
        data = ImageDataBunch.from_folder(path, test='test', valid_pct=valid_pct)
        mnist_tiny_sanity_test(data)

def test_from_name_re(path):
    fnames = get_files(path/'train', recurse=True)
    pat = r'\/([^/]+)\/\d+.png$'
    data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=(rand_pad(2, 28), []))
    mnist_tiny_sanity_test(data)

def test_from_csv(path):
    files = []
    for each in ['train', 'valid', 'test']: files += get_files(path/each, recurse=True)
    tmp_path = path/'tmp'
    try:
        os.makedirs(tmp_path)
        for filepath in files: shutil.copyfile(filepath, tmp_path/filepath.name)
        prep_mnist_tiny_csv_labels(path).to_csv(tmp_path/'labels.csv', index=False)
        data = ImageDataBunch.from_csv(tmp_path, size=28)
        mnist_tiny_sanity_test(data)
    finally:
        shutil.rmtree(tmp_path)