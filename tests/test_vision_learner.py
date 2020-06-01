from efficientnet_pytorch import EfficientNet
import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet18
from fastai.datasets import untar_data, URLs
from fastai.gen_doc.doctest import this_tests
from fastai.vision.learner import *
from fastai.callbacks.hooks import *
from fastai.vision import ImageDataBunch, Learner
from fastai.vision.models import EfficientNetB1
from fastai.vision.learner import has_pool_type

@pytest.fixture
def image():
    return torch.randn([4, 3, 32, 32])


def test_create_body(image):
    this_tests(create_body)
    def get_hook_fn(actns):
        return lambda self,input,output: actns.append(output)
    def run_with_capture(m):
        actns = []
        hooks = Hooks(m, get_hook_fn(actns))
        m(image)
        hooks.remove()
        return actns
    body = create_body(resnet18, pretrained=True, cut=-2)
    resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
    body_actns = run_with_capture(body)
    resnet_actns = run_with_capture(resnet)
    for i in range(len(body_actns)):
        assert torch.allclose(body_actns[i], resnet_actns[i]) # check activation values at each block

    body = create_body(resnet18, cut=lambda x:x)
    assert isinstance(body, type(resnet18()))

    with pytest.raises(NameError):
        create_body(resnet18, cut=1.)

def test_create_head(image):
    this_tests(create_head)
    nc = 4 # number of output classes
    head = create_head(nf=image.shape[1]*2,nc=nc)
    assert list(head(image).shape) == [image.shape[0],nc]

def test_has_pool_type():
	this_tests(has_pool_type)
	nc = 5 # dummy number of output classes
	rn18m = create_cnn_model(resnet18, nc=nc)
	assert has_pool_type(rn18m) # rn34 has pool type


@pytest.mark.parametrize("image_size", [128, 224, 256])
def test_create_body_effnet(image_size):
    this_tests(create_body)
    base_arch = EfficientNetB1
    imgs = torch.randn([4, 3, image_size, image_size])
    body = create_body(base_arch, pretrained=True)
    ref = EfficientNet.from_pretrained(f"efficientnet-b{base_arch.__name__[-1]}")
    body.eval()
    ref.eval()
    assert torch.allclose(body(imgs), ref.extract_features(imgs)) # check activation values after conv blocks

    body = create_body(base_arch, cut=lambda x:x)
    assert isinstance(body, type(base_arch()))

    with pytest.raises(NameError):
        create_body(base_arch, cut=1.)

def test_freeze_unfreeze_effnet():
    this_tests(cnn_learner)
    def get_number_of_trainable_params(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_arch = EfficientNetB1
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path, size=64)
    data.c = 1000  # Set number of class to be 1000 to stay in line with the pretrained model.
    cnn_learn = cnn_learner(data, base_arch, pretrained=True)
    ref_learn = Learner(data, EfficientNet.from_pretrained("efficientnet-b1"))
    # By default the neural net in cnn learner is freezed.
    assert get_number_of_trainable_params(cnn_learn.model) < get_number_of_trainable_params(ref_learn.model)
    cnn_learn.unfreeze()
    assert get_number_of_trainable_params(cnn_learn.model) == get_number_of_trainable_params(ref_learn.model)
