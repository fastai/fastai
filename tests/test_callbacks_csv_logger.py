import pytest
from fakes import *
from io import StringIO
from contextlib import redirect_stdout

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
    lines = buffer.getvalue().split('\n')[:-1]
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

def test_logger():
    learn = fake_learner()
    learn.metrics = [exp_rmspe]
    learn.callback_fns.append(callbacks.CSVLogger)
    buffer = StringIO()
    with redirect_stdout(buffer): learn.fit_one_cycle(3)
    csv_df = learn.csv_logger.read_logged_file()
    recorder_df = create_metrics_dataframe(learn)
    pd.testing.assert_frame_equal(csv_df, recorder_df, check_exact=False, check_less_precise=True)
    stdout_df = convert_into_dataframe(buffer)
    pd.testing.assert_frame_equal(csv_df, stdout_df, check_exact=False, check_less_precise=True)
    