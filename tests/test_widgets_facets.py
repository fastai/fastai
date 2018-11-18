import PIL.Image
import json
import os.path
import torch
from torch import Tensor
from fastai.metrics import accuracy
from fastai.vision import ImageDataBunch, create_cnn, models
from fastai.vision.data import imagenet_stats, get_transforms
from fastai.widgets import FacetsDive
from fastai.imports import Path

torch.manual_seed(1729)

fake_preds = [
    Tensor([[0.2023, 0.1316, 0.0330, 0.1373, 0.4959],
            [0.2873, 0.0384, 0.2014, 0.1811, 0.2918],
            [0.2485, 0.1279, 0.0741, 0.2880, 0.2615],
            [0.2809, 0.0556, 0.0168, 0.0410, 0.6058],
            [0.5330, 0.0334, 0.1005, 0.0661, 0.2670],
            [0.3911, 0.0442, 0.0781, 0.1405, 0.3462],
            [0.1110, 0.1209, 0.0418, 0.0937, 0.6326],
            [0.1314, 0.2569, 0.0069, 0.0418, 0.5630],
            [0.1273, 0.0945, 0.0138, 0.0173, 0.7470],
            [0.3101, 0.0493, 0.0261, 0.0936, 0.5208],
            [0.0603, 0.0780, 0.0231, 0.0886, 0.7500],
            [0.0374, 0.4304, 0.0394, 0.0509, 0.4419],
            [0.1648, 0.0418, 0.0079, 0.0458, 0.7396],
            [0.0262, 0.0502, 0.0131, 0.0756, 0.8349],
            [0.0920, 0.0692, 0.0201, 0.0991, 0.7196]]),
    Tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
    Tensor([1.5982, 1.2473, 1.3922, 2.8897, 3.3988, 3.1196, 3.1745, 4.9806,
            4.2795, 2.3685, 2.4235, 2.9774, 0.3016, 0.1805, 0.3291])]

expected_json_keys = {
    "aspect_ratio", "class_idx", "class_name", "height", "is_pred_accurate", "is_vertical", "loss", "pred_class_idx",
    "pred_class_name", "prob", "res", "sample_idx", "sample_name", "thumb_height", "thumb_width", "width"}


def test_facets_dive_no_preds(tmp_path):
    folder = 'data/test/images/facets/bird_species/'
    # pat = re.compile(r'(\d+\.\w+)\/\w+\.jpg$')
    data = ImageDataBunch.from_folder(folder, ds_tfms=get_transforms(), size=224, bs=64)
    data.normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34, metrics=accuracy)
    preds = learn.get_preds(with_loss=True)

    viz = FacetsDive(data, preds=preds, facets_path=tmp_path)
    viz.show()

    # Check that preds attribute is converted to lists
    assert all(isinstance(x, list) for x in viz.preds)
    assert len(viz.preds) == 3

    # Check that all the keys are there in the metadata
    assert all(not (expected_json_keys-set(x.keys())) for x in viz.metadata)


def test_facets_dive_with_preds_no_filters(tmp_path):
    folder = 'data/test/images/facets/bird_species/'
    data = ImageDataBunch.from_folder(folder, ds_tfms=get_transforms(), size=224, bs=64)
    data.normalize(imagenet_stats)

    viz = FacetsDive(data, preds=fake_preds, facets_path=tmp_path)
    viz.show()

    # Check that the file is generated in the right folder
    assert Path(os.path.dirname(viz.path_sprite)) == tmp_path
    # Check that image is of the right size
    with PIL.Image.open(viz.path_sprite) as im:
        im_size = im.size
    assert im_size == (128*3, 128*5)
    assert all(not (expected_json_keys-set(x.keys())) for x in viz.metadata)


def test_facets_dive_with_preds_and_filters(tmp_path):
    folder = 'data/test/images/facets/bird_species/'
    data = ImageDataBunch.from_folder(folder, ds_tfms=get_transforms(), size=224, bs=64)
    data.normalize(imagenet_stats)

    filter_fn = lambda **kwargs: kwargs['loss'] > 2 and kwargs['class_name'] != '005.Crested_Auklet'
    viz = FacetsDive(data, preds=fake_preds, filter_fn=filter_fn, facets_path=tmp_path)
    viz.show()

    # Check that the file is generated in the right folder
    assert Path(os.path.dirname(viz.path_sprite)) == tmp_path
    with PIL.Image.open(viz.path_sprite) as im:
        im_size = im.size
    assert im_size == (128*3, 128*3)
    assert all(not (expected_json_keys-set(x.keys())) for x in viz.metadata)
