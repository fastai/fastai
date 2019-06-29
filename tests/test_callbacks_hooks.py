import pytest, torch, fastai
from fastai.gen_doc.doctest import this_tests
from fastai.basics import *
from fastai.callbacks import *
from fastai.callbacks.hooks import *
from fastai.vision import *
from fastai.text import *
from fastai.tabular import *
from fastai.collab import *

use_gpu = torch.cuda.is_available()

@pytest.fixture(scope="module")
def mnist_path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def test_model_summary_vision(mnist_path):
    this_tests(model_summary)
    path = mnist_path
    data = ImageDataBunch.from_folder(path, ds_tfms=([], []), bs=2)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    _ = model_summary(learn)

@pytest.mark.xfail(reason = "Expected Fail, text models not supported yet.")
def test_model_summary_text():
    this_tests(model_summary)
    path = untar_data(URLs.IMDB_SAMPLE)
    data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
    learn = language_model_learner(data_lm, pretrained_model=None)
    _ = model_summary(learn)

def test_model_summary_tabular():
    this_tests(model_summary)
    path = untar_data(URLs.ADULT_SAMPLE)
    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    procs = [FillMissing, Categorify]
    df = pd.read_csv(path/'adult.csv')
    data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                               .split_by_idx(list(range(800,1000)))
                               .label_from_df(cols=dep_var)
                               .databunch(bs=2))
    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
    _ = model_summary(learn)

def test_model_summary_collab():
    this_tests(model_summary)
    path = untar_data(URLs.ML_SAMPLE)
    ratings = pd.read_csv(path/'ratings.csv')
    series2cat(ratings, 'userId', 'movieId')
    data = CollabDataBunch.from_df(ratings, seed=42, bs=2)
    y_range = [0,5.5]
    learn = collab_learner(data, n_factors=50, y_range=y_range)
    _ = model_summary(learn)

#model_summary takes a Learner now
#def test_model_summary_nn_module():
#    _ = model_summary(nn.Conv2d(16,16,3,padding=1))
#
#def test_model_summary_nn_modules():
#    class BasicBlock(Module):
#        def __init__(self):
#            super().__init__()
#            self.conv1 = conv2d(16,16,3,1)
#            self.conv2 = conv2d(16,16,3,1)
#        def forward(self, x):
#            x = self.conv1(x)
#            x = self.conv2(x)
#            return x
#    _ = model_summary(BasicBlock())

def test_hook_output_basics(mnist_path):
    this_tests(hook_output)
    data = ImageDataBunch.from_folder(mnist_path, size=128, bs=2)
    learn = cnn_learner(data, models.resnet18)
    # need to train to get something meaningful, but for just checking shape its fine w/o it
    m = learn.model.eval()
    x,y = data.train_ds[0]
    xb,_ = data.one_item(x)
    if use_gpu: xb = xb.cuda()

    def hooked(cat=y):
        with hook_output(m[0]) as hook_forward:
            preds = m(xb)
        with hook_output(m[0]) as hook_backward:
            preds = m(xb)
            preds[0,int(cat)].backward()
        return hook_forward, hook_backward

    for hook in hooked():
        acts = hook.stored[0].cpu()
        assert list(acts.shape) == [512, 4, 4], "activations via hooks"
