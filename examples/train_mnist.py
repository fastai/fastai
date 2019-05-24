from fastai.script import *
from fastai.vision import *
from fastai.distributed import *

@call_parse
def main():
    path = url2path(URLs.MNIST_SAMPLE)
    tfms = (rand_pad(2, 28), [])
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=64).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit_one_cycle(1, 0.02)

