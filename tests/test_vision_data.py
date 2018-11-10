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

def test_from_csv_and_from_df(path):
    for func in ['from_csv', 'from_df']:
        files = []
        for each in ['train', 'valid', 'test']: files += get_files(path/each, recurse=True)
        tmp_path = path/'tmp'
        try:
            os.makedirs(tmp_path)
            for filepath in files: shutil.copyfile(filepath, tmp_path/filepath.name)
            if func is 'from_df': data = ImageDataBunch.from_df(tmp_path, df=pd.read_csv(path/'labels.csv'), size=28)
            else:
                shutil.copyfile(path/'labels.csv', tmp_path/'labels.csv')
                data = ImageDataBunch.from_csv(tmp_path, size=28)
            mnist_tiny_sanity_test(data)
        finally:
            shutil.rmtree(tmp_path)

def test_from_df_test_dataset(path):
    "Check that test dataset is created with from_df."
    #Actual contents do not matter here.
    df = pd.DataFrame({'fn': ['a.jpg', 'b.jpg', 'c.jpg'],
                       'lbl': [0, 1, 2]})
    data = ImageDataBunch.from_df('.', df, path, test='test')
    #If the test set is not registered, this will raise an assertion error
    data.test_ds

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