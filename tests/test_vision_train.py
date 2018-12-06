import pytest
from fastai import *
from fastai.vision import *
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.integration

def create_metrics_dataframe(learn):
    "Converts metrics stored in `Recorder` into dataframe."
    records = [
        [i, loss, val_loss, *itemize(epoch_metrics)]
        for i, (loss, val_loss, epoch_metrics)
        in enumerate(zip(
            get_train_losses(learn),
            learn.recorder.val_losses,
            learn.recorder.metrics), 1)]
    return pd.DataFrame(records, columns=learn.recorder.names)

def convert_into_dataframe(buffer):
    "Converts data captured from `fastprogress.ConsoleProgressBar` into dataframe."
    lines = buffer.getvalue().split('\n')
    header, *lines = [l.strip() for l in lines if l]
    header = header.split()
    floats = [[float(x) for x in line.split()] for line in lines]
    records = [dict(zip(header, metrics_list)) for metrics_list in floats]
    df = pd.DataFrame(records, columns=header)
    df['epoch'] = df['epoch'].astype(int)
    return df

def get_train_losses(learn):
    "Returns list of training losses at the end of each training epoch."
    np_losses = [to_np(l).item() for l in learn.recorder.losses]
    batch_size = len(learn.data.train_dl)
    return [batch[-1] for batch in partition(np_losses, batch_size)]

def itemize(metrics):
    return [m.item() for m in metrics]

@pytest.fixture
def no_bar():
    fastprogress.NO_BAR = True
    yield
    fastprogress.NO_BAR = False

@pytest.fixture(scope="module")
def learn():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), batch_size=16, num_workers=2)
    data.normalize()
    learn = Learner(data, simple_cnn((3,16,16,16,2), bn=True), metrics=[accuracy, error_rate],
                                 callback_fns=[callbacks.CSVLogger])
    buffer = StringIO()
    with redirect_stdout(buffer): learn.fit_one_cycle(3)
    csv_df = learn.csv_logger.read_logged_file()
    recorder_df = create_metrics_dataframe(learn)
    pd.testing.assert_frame_equal(csv_df, recorder_df, check_exact=False, check_less_precise=True)
    stdout_df = convert_into_dataframe(buffer)
    pd.testing.assert_frame_equal(csv_df, stdout_df, check_exact=False, check_less_precise=True)
    return learn

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
        if outputs[int(label)] > outputs[1-int(label)]: return
    assert False, 'Failed to predict correct class'

def test_interp(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_interp_shortcut(learn):
    interp = learn.interpret()
    losses,idxs = interp.top_losses()
    assert len(learn.data.valid_ds)==len(losses)==len(idxs)

def test_lrfind(learn):
    learn.lr_find(start_lr=1e-5,end_lr=1e-3, num_it=15)
