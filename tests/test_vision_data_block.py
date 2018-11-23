import pytest
from fastai import *
from fastai.vision import *

def _print_data(data): print(len(data.train_ds),len(data.valid_ds))
def _check_data(data, t, v):
    assert len(data.train_ds)==t
    assert len(data.valid_ds)==v
    _ = data.train_ds[0]

def test_vision_datasets():
    il = ImageItemList.from_folder(untar_data(URLs.MNIST_TINY))
    sds = il.split_by_idx([0]).label_from_folder().add_test_folder()
    assert np.array_equal(sds.train.classes, sds.valid.classes), 'train/valid classes same'
    assert len(sds.test)==20, "test_ds is correct size"
    data = sds.databunch()
    _check_data(data, len(il)-1, 1)

def test_multi():
    path = untar_data(URLs.PLANET_TINY)
    data = (ImageItemList.from_csv(path, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct(seed=42).label_from_df(sep=' ').databunch())
    x,y = data.valid_ds[0]
    assert x.shape[0]==3
    assert data.c==len(y.data)==14
    assert len(str(y))>2
    _check_data(data, 160, 40)

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
    _check_data(data, 80, 20)

def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)

def test_points():
    coco = untar_data(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(coco/'train.json')
    points = [tensor([b[0][0][0], b[0][0][1]]) for b in lbl_bbox]
    img2pnts = dict(zip(images, points))
    get_y_func = lambda o:img2pnts[o.name]
    data = (ImageItemList.from_folder(coco)
            .random_split_by_pct()
            .label_from_func(get_y_func, label_cls=PointsItemList)
            .transform(get_transforms(), tfm_y=True)
            .databunch())
    _check_data(data,160,40)

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
    _check_data(data, 160, 40)

def test_image_to_image_different_y_size():
    get_y_func = lambda o:o
    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms()
    data = (ImageItemList.from_folder(mnist)
            .random_split_by_pct()
            .label_from_func(get_y_func)
            .transform(tfms, size=20)
            .transform_y(size=80)
            .databunch(bs=16))

    x,y = data.one_batch()
    assert x.shape[2]*4 == y.shape[3]

def test_image_to_image_different_tfms():
    get_y_func = lambda o:o
    mnist = untar_data(URLs.COCO_TINY)
    x_tfms = get_transforms()
    y_tfms = [[t for t in x_tfms[0]], [t for t in x_tfms[1]]]
    y_tfms[0].append(flip_lr())
    data = (ImageItemList.from_folder(mnist)
            .random_split_by_pct()
            .label_from_func(get_y_func)
            .transform(x_tfms)
            .transform_y(y_tfms)
            .databunch(bs=16))

    x,y = data.one_batch()
    x1 = x[0]
    y1 = y[0]
    x1r = flip_lr(Image(x1)).data
    assert (y1 == x1r).all()

