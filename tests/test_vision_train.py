import pytest
from fastai.vision import *
from fastai.gen_doc.doctest import this_tests
from fastai.callbacks import *
from fastai.utils.mem import *
from utils.mem import *
from math import isclose
from fastai.train import ClassificationInterpretation

use_gpu = torch.cuda.is_available()
torch_preload_mem()

pytestmark = pytest.mark.integration

@pytest.fixture
def no_bar():
    fastprogress.NO_BAR = True
    yield
    fastprogress.NO_BAR = False

@pytest.fixture(scope="module")
def mnist_tiny():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), num_workers=2)
    data.normalize()
    return data

@pytest.fixture(scope="module")
def zero_image():
    return Image(torch.zeros((3, 128, 128)))

@pytest.fixture(scope="module")
def learn(mnist_tiny):
    # path = untar_data(URLs.MNIST_TINY)
    # data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), num_workers=2)
    # data.normalize()
    learn = Learner(mnist_tiny, simple_cnn((3,16,16,16,2), bn=True), metrics=[accuracy, error_rate])
    learn.fit_one_cycle(3)
    return learn

def test_1cycle_lrs(learn):
    lrs = learn.recorder.lrs
    this_tests(learn.recorder.__class__)
    assert lrs[0]<0.001
    assert lrs[-1]<0.0001
    assert np.max(lrs)==3e-3

def test_1cycle_moms(learn):
    this_tests(learn.recorder.__class__)
    moms = learn.recorder.moms
    assert moms[0]==0.95
    assert abs(moms[-1]-0.95)<0.01
    assert np.min(moms)==0.85
    
def test_accuracy(learn):
    this_tests(accuracy)
    assert accuracy(*learn.get_preds()) > 0.9

def test_error_rate(learn):
    this_tests(error_rate)
    assert error_rate(*learn.get_preds()) < 0.1

def test_preds(learn):
    this_tests(learn.predict)
    pass_tst = False
    for i in range(3):
        img, label = learn.data.valid_ds[i]
        pred_class,pred_idx,outputs = learn.predict(img)
        if outputs[int(label)] > outputs[1-int(label)]: return
    assert False, 'Failed to predict correct class'

def test_interp(learn):
    this_tests(ClassificationInterpretation.from_learner)
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_interp_shortcut(learn):
    this_tests(learn.interpret)
    interp = learn.interpret()
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_lrfind(learn):
    this_tests(learn.lr_find)
    learn.lr_find(start_lr=1e-5,end_lr=1e-3, num_it=15)

@pytest.mark.parametrize('arch', [models.resnet18, models.squeezenet1_1])
def test_models_meta(mnist_tiny, arch, zero_image):
    learn = cnn_learner(mnist_tiny, arch, metrics=[accuracy, error_rate])
    this_tests(learn.predict)
    pred = learn.predict(zero_image)
    assert pred is not None

def test_ClassificationInterpretation(learn):
    this_tests(ClassificationInterpretation)
    interp = ClassificationInterpretation.from_learner(learn)
    assert isinstance(interp.confusion_matrix(), (np.ndarray))
    assert interp.confusion_matrix().sum() == len(learn.data.valid_ds)
    conf = interp.most_confused()
    expect = {'3', '7'}
    assert (len(conf) == 0 or
            len(conf) == 1 and (set(conf[0][:2]) == expect) or
            len(conf) == 2 and (set(conf[0][:2]) == set(conf[1][:2]) == expect)
    ), f"conf={conf}"
