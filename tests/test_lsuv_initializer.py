import pytest

from fastai.core import VV
from fastai.lsuv_initializer import apply_lsuv_init

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models


@pytest.fixture
def image_data():
    images_to_process = []
    for img_fname in os.listdir('fastai/images'):
        img = cv2.imread(os.path.join('fastai/images', img_fname))
        images_to_process.append(np.transpose(cv2.resize(img, (224,224)), (2,0,1)))
    data = np.array(images_to_process).astype(np.float32)
    return VV(torch.from_numpy(data)).cpu()


def add_hooks(m, fn):
    hooks = []
    def add_hook(m):
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            hooks.append(m.register_forward_hook(fn))
    m.apply(add_hook)
    return hooks
def remove_hooks(hooks): [h.remove() for h in hooks]

def run_with_capture(m, data):
    activation_variances = []
    def capture_hook(self, input, output):
        activation_variances.append(np.var(output.data.cpu().numpy()))
    hooks = add_hooks(m, capture_hook)
    m(data)
    remove_hooks(hooks)
    return activation_variances

def test_fast_initialization_without_orthonormal(image_data):
    alexnet = models.alexnet(pretrained=False)
    pre_init_var = run_with_capture(alexnet, image_data)
    assert pre_init_var[0] >= 1000  # the first few pre-init variances are huge,
    assert pre_init_var[1] >= 100   # even larger than these conservative tests.

    tol = 0.1
    alexnet = apply_lsuv_init(alexnet, image_data, std_tol=tol, do_orthonorm=False, cuda=False)
    *post_init_var, final_var = run_with_capture(alexnet, image_data)
    for var in post_init_var:
        assert 2 <= var <= 4
    assert final_var == pytest.approx(1, tol**2)
