import pytest
import torch
import torch.nn as nn
from fastai.gen_doc.doctest import this_tests
from fastai.vision.learner import *
from fastai.callbacks.hooks import *
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from fastai.datasets import *
from fastai.vision.data import *
from fastai.vision import *

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


def test_yolo_learner():
    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    data = load_coco('./coco_dataset/', './coco_dataset/annotations/train.json',
                     './coco_dataset/annotations/valid.json', tfms=tfms)
    '''
    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms(do_flip=False)
    data = (ImageList.from_folder(mnist)
            .split_by_folder()
            .label_from_folder()
            .transform(tfms, size=32)
            .databunch()
            .normalize(imagenet_stats))
    '''
    learn = yolo_learner(data, metrics=accuracy)
    preds, y = learn.get_preds()

    #learner.fit_one_cycle(1,1e-2)
    #learner.show_results()


