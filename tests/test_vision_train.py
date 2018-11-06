import pytest
from fastai import *
from fastai.vision import *

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
    data.normalize()
    learn = ClassificationLearner(data, simple_cnn((3,16,16,16,2), bn=True), metrics=accuracy)
    learn.fit_one_cycle(3)
    return learn

def _run_batch_size_test(bs,nfiles):
    np.random.seed(0)
    path = untar_data(URLs.MNIST_TINY)
    dummy_fnames = [str(x.relative_to(path)) for x in (path / 'train' / '3').glob('**/*.png')]
    dummy_data = {'fn': dummy_fnames[:nfiles], 'label': ['3'] * (nfiles)}
    dummy_df = pd.DataFrame(dummy_data)
    data = ImageDataBunch.from_df(path, dummy_df, size=224, bs=bs, num_workers=2, valid_pct=0.5)
    data.normalize()
    learn = create_cnn(data, models.resnet18, metrics=accuracy, pretrained=False)
    learn.fit_one_cycle(1)

# results in training batches with size 4 and 1
def test_batch_size_4():
    bs=4
    nfiles=6
    _run_batch_size_test(bs,nfiles)

# results in training batches with size 3 and 2
def test_batch_size_3():
    bs=3
    nfiles=6
    _run_batch_size_test(bs,nfiles)

def test_accuracy(learn):
    assert accuracy(*learn.get_preds()) > 0.9

def test_error_rate(learn):
    assert error_rate(*learn.get_preds()) < 0.1

def test_1cycle_lrs(learn):
    lrs = learn.recorder.lrs
    assert lrs[0]<0.001
    assert lrs[-1]<0.0001
    assert np.max(lrs)==3e-3

def test_1cycle_moms(learn):
    moms = learn.recorder.moms
    assert moms[0]==0.95
    assert abs(moms[-1]-0.95)<0.01
    assert np.min(moms)==0.85

def test_preds(learn):
    pass_tst = False
    for i in range(3):
        img, label = learn.data.valid_ds[i]
        pred_class,pred_idx,outputs = learn.predict(img)
        if outputs[label] > outputs[1-label]: return
    assert False, 'Failed to predict correct class'

def test_lrfind(learn):
    learn.lr_find(start_lr=1e-5,end_lr=1e-3, num_it=15)
