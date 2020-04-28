import pytest, torch
from fastai.gen_doc.doctest import this_tests
from utils.text import CaptureStdout
from fastai.datasets import untar_data, URLs
from fastai.core import noop
from fastai.vision.gan import *


@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

@pytest.fixture(scope="module")
def data(path):
    data = (GANItemList.from_folder(path, noise_sz=5)
                       .split_none()
                       .label_from_func(noop)
                       .transform(size=32, tfm_y=True) # image size needs to be a power of 2
                       .databunch(bs=16))
    return data

@pytest.fixture(scope="module")
def gan_learner(data):
    generator = basic_generator(32, 3, 5)
    critic = basic_critic(32, 3, 16)
    return GANLearner.wgan(data, generator, critic)


def test_gan_datasets(path):
    this_tests(GANItemList.from_folder)
    lls = GANItemList.from_folder(path).split_none().label_from_func(noop)

    assert len(lls.train) == 1428
    assert isinstance(lls.train.x, GANItemList)

def test_noisy_item():
    this_tests(NoisyItem)
    item = NoisyItem(10)

    assert item.obj == 10
    assert item.data.size() == torch.Size([10, 1, 1])
    assert f"{item}" == ""

def test_basic_generator():
    this_tests(basic_generator)

    batch_size = 2; noise_size = 10; img_size = 16; n_channels = 3; n_features = 8;
    noise = torch.randn((batch_size, noise_size, 1, 1))
    generator = basic_generator(img_size, n_channels, noise_size, n_features)

    out = generator(noise)
    assert out.size() == torch.Size([batch_size, n_channels, img_size, img_size])

def test_basic_critic():
    this_tests(basic_critic)

    batch_size = 2; img_size = 16; n_channels = 3; n_features = 8;
    image = torch.randn((batch_size, n_channels, img_size, img_size))
    critic = basic_critic(img_size, n_channels, n_features)

    out = critic(image)
    assert out.size() == torch.Size([1])


def test_gan_module(data):
    this_tests(GANModule)
    generator = basic_generator(32, 3, 5, 6)
    critic = basic_critic(32, 3)
    gan_module = GANModule(generator, critic, gen_mode=True)
    noise, image = data.one_batch()

    assert isinstance(gan_module(noise), torch.Tensor)
    gan_module.switch()
    assert gan_module.gen_mode == False
    assert isinstance(gan_module(image), torch.Tensor)

@pytest.mark.slow
def test_gan_trainer(gan_learner):
    this_tests(GANTrainer)
    gan_trainer = gan_learner.gan_trainer
    with CaptureStdout() as cs: gan_learner.fit(1, 1e-4)
    assert gan_trainer.imgs
    assert gan_trainer.gen_mode
    assert gan_trainer.titles

