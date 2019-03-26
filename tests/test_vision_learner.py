import pytest
import torch
from fastai.gen_doc.doctest import this_tests
from fastai.vision.learner import *
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

@pytest.fixture
def image():
    return torch.randn([4, 3, 32, 32])


def add_hooks(m, fn):
    hooks = []
    def add_hook(m):
        if isinstance(m, BasicBlock):
            hooks.append(m.register_forward_hook(fn))
    m.apply(add_hook)
    return hooks

def remove_hooks(hooks):
    for h in hooks: h.remove()

def run_with_capture(m, image):
    activations = []
    def capture_hook(self, input, output):
        activations.append(output)
    hooks = add_hooks(m, capture_hook)
    m(image)
    remove_hooks(hooks)
    return activations

def test_create_body(image):
    this_tests(create_body)
    body = create_body(resnet18, pretrained=True, cut=-2).eval()
    model = resnet18(pretrained=True).eval()
    body_actns = run_with_capture(body, image)
    model_actns = run_with_capture(model, image)
    n = len(body_actns) 
    for i in range(n):
        assert torch.allclose(body_actns[i], model_actns[i])

def test_create_head(image):
    this_tests(create_head)
    nc = 4 # number of output classes
    head = create_head(nf=image.shape[1]*2,nc=nc)
    assert list(head(image).shape) == [image.shape[0],nc]
