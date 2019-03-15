import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.vision.models import *
from fastai.vision.learner import create_body
import torch.nn as nn
import torch

@pytest.fixture
def model():
    body = create_body(resnet18, pretrained=False)
    for param in body.parameters():
        param.requires_grad = False
    return DynamicUnet(body, 10)

@pytest.fixture
def image():
    return torch.randn([4,3,32,32]) # create fake image


def add_hooks(m, fn):
    hooks = []
    def add_hook(m):
        if isinstance(m, UnetBlock):
            hooks.append(m.register_forward_hook(fn))
    m.apply(add_hook)
    return hooks

def remove_hooks(hooks): [h.remove() for h in hooks]

def run_with_capture(m, image):
    activation_shapes = []
    def capture_hook(self, input, output):
        activation_shapes.append(output.shape)
    hooks = add_hooks(m, capture_hook)
    m(image)
    remove_hooks(hooks)
    return activation_shapes

def test_dynamic_unet_shape(model, image):
    this_tests(DynamicUnet)
    pred = model(image)
    assert list(pred.shape[-2:]) == [32,32] # image HxW should remain the same
    assert pred.shape[1] == 10 # number of output classes 

def test_unet_block_shapes(model, image):
    this_tests(DynamicUnet)
    expected_shapes = [[4,512,2,2],[4,384,4,4],[4,256,8,8],[4,96,16,16]]
    activation_shapes = run_with_capture(model, image)
    for act, exp in zip(activation_shapes, expected_shapes):
        assert list(act) == exp
