import pytest
import torch
import torch.nn as nn
from fastai.gen_doc.doctest import this_tests
from fastai.vision.learner import *
from fastai.callbacks.hooks import *
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
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
