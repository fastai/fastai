import pytest
from fastai import *
from fastai.vision import *

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def mnist_tiny_sanity_test(data):
    assert data.c == 2
    assert set(map(str, set(data.classes))) == {'3', '7'}

def test_from_folder(path):
    for valid_pct in [None, 0.9]:
        data = ImageDataBunch.from_folder(path, test='test')
        mnist_tiny_sanity_test(data)

def test_from_name_re(path):
    fnames = get_files(path/'train', recurse=True)
    pat = r'\/([^/]+)\/\d+.png$'
    data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=(rand_pad(2, 28), []))
    mnist_tiny_sanity_test(data)

def test_from_csv_and_from_df(path):
    for func in ['from_csv', 'from_df']:
        files = []
        if func is 'from_df': data = ImageDataBunch.from_df(path, df=pd.read_csv(path/'labels.csv'), size=28)
        else: data = ImageDataBunch.from_csv(path, size=28)
        mnist_tiny_sanity_test(data)

def test_download_images():
    base_url = 'http://files.fast.ai/data/tst_images/'
    fnames = ['tst0.jpg', 'tst1.png', 'tst2.tif']

    tmp_path = URLs.LOCAL_PATH/'data'/'tmp'
    try:
        os.makedirs(tmp_path)
        with open(tmp_path/'imgs.txt', 'w') as f:
            [f.write(f'{base_url}{fname}\n') for fname in fnames]
        download_images(tmp_path/'imgs.txt', tmp_path)
        for fname in fnames:
            ext = fname.split('.')[-1]
            files = list(tmp_path.glob(f'*.{ext}'))
            assert len(files) == 1
            assert os.path.getsize(files[0]) > 0
    finally:
        shutil.rmtree(tmp_path)
