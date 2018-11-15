import pytest
from fastai import *
from fastai.vision import *

def test_vision_datasets():
    sds = (ImageItemList.from_folder(untar_data(URLs.MNIST_TINY))
           .split_by_idx([0])
           .label_from_folder()
           .add_test_folder())
    assert np.array_equal(sds.train.classes, sds.valid.classes), 'train/valid classes same'
    assert len(sds.test)==20, "test_ds is correct size"

def test_multi():
    path = untar_data(URLs.PLANET_TINY)
    data = (ImageItemList.from_csv(path, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct().label_from_df(sep=' ').databunch())
    x,y = data.valid_ds[0]
    assert x.shape[0]==3
    assert data.c==len(y.data)==14
    assert len(str(y))>2         

def test_camvid():
    camvid = untar_data(URLs.CAMVID_TINY)
    path_lbl = camvid/'labels'
    path_img = camvid/'images'
    codes = np.loadtxt(camvid/'codes.txt', dtype=str)
    get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
    data = (SegmentationItemList.from_folder(path_img)
            .random_split_by_pct()
            .label_from_func(get_y_fn, classes=codes)
            .transform(get_transforms(), tfm_y=True)
            .databunch())

def test_coco():
    coco = untar_data(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(coco/'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o:img2bbox[o.name]
    data = (ObjectItemList.from_folder(coco)
            .random_split_by_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=16, collate_fn=bb_pad_collate))
