import pytest
from fastai import *
from fastai.vision import *

def bbox2pic(corners, size):
    pic = torch.zeros(1, size,size)
    pic[0, corners[0]:corners[2], corners[1]:corners[3]] = 1.
    return Image(pic)

def points2pic(points, size):
    points = points.clamp(min=0,max=size-1)
    pic = torch.zeros(1, size,size)
    for p in points: pic[0, max(0,int(p[0])-1):min(size, int(p[0])+1),max(0,int(p[1])-1):min(size, int(p[1])+1)] = 1.
    return Image(pic)

def create_data(img, target, size, **kwargs):
    ll = LabelList(ItemList([img]), ItemList([target]))
    lls = LabelLists(Path('.'), ll, ll)
    lls = lls.transform(get_transforms(), size=size, **kwargs)
    return lls

def test_points_data_aug():
    "Check that ImagePoints get changed with their input Image."
    points = torch.randint(0,64, ((5,2)))
    img = points2pic(points, 64)
    pnts = ImagePoints(FlowField((64,64), points.float()))
    lls = create_data(img, pnts, 64, mode='nearest')
    tfm_x,tfm_y = lls.train[0]
    new_pnts = scale_flow(FlowField(tfm_y.size, tfm_y.data), to_unit=False).flow.round()
    fail = False
    for p in new_pnts.round():
        if tfm_x.data[0, max(0,int(p[0])-1):min(int(p[0])+2,64), max(0,int(p[1])-1):min(int(p[1])+2,64)].sum() < 0.8:
            fail = True
    assert not fail

def test_bbox_data_aug():
    "Check that ImagePoints get changed with their input Image."
    pick_box = True
    while pick_box:
        points = torch.randint(5,59, ((2,2)))
        #Will fail if box to close to the border
        corners = torch.cat([points.min(0)[0], points.max(0)[0]])
        pick_box = (corners[2:] - corners[:2]).min() < 2
    img = bbox2pic(corners, 64)
    bbox = ImageBBox.create(64, 64, [list(corners)])
    lls = create_data(img, bbox, 64, mode='nearest', padding_mode='zeros')
    tfm_x,tfm_y = lls.train[0]
    new_bb = ((tfm_y.data + 1) * 32)
    mask = (tfm_x.data[0] > 0.5).nonzero()
    if len(mask) == 0:
        assert (new_bb[0][2:] - new_bb[0][:2]).min() < 1
    else:
        img_bb = torch.cat([mask.min(0)[0], mask.max(0)[0]])
        assert (new_bb - img_bb.float()).abs().max() < 2

def test_mask_data_aug():
    points = torch.randint(0,2, ((1,64,64))).float()
    img, mask = Image(points), ImageSegment(points)
    lls = create_data(img, mask, 64, mode='nearest')
    tfm_x,tfm_y = lls.train[0]
    new_mask = (tfm_x.data[0] > 0.5)
    assert (new_mask.float() - tfm_y.data[0].float()).sum() < 1.

def img_test(cs):
    points = torch.zeros(5,5)
    if not is_listy(cs[0]): cs = [cs]
    for c in cs: points[c[0],c[1]] = 1
    return Image(points[None])

def check_image(x, cs):
    if not is_listy(cs[0]): cs = [cs]
    target = torch.zeros(*x.size)
    for c in cs: target[c[0],c[1]] = 1
    assert (x.data - target).abs().sum() <5e-7

def check_tfms(img, tfms, targets, **kwargs):
    for tfm, t in zip(tfms, targets):
        check_image(img.apply_tfms(tfm, **kwargs), t)

def test_all_warps():
    signs = [1,1,1,-1,-1,1,-1,-1]
    inputs = [[0,0], [0,0], [4,0], [4,0], [0,4], [0,4], [4,4], [4,4]]
    targets = [[0,1], [1,0], [4,1], [3,0], [0,3], [1,4], [4,3], [3,4]]
    for k, (i,t, s) in enumerate(zip(inputs, targets, signs)):
        magnitudes = torch.zeros(8)
        magnitudes[k] = s * 0.5
        check_image(perspective_warp(img_test(i), magnitude=magnitudes), t)
        tfm = [skew(magnitude=-0.5)]
        tfm[0].resolved = {'direction':k, 'magnitude':-0.5}
        check_image(img_test(i).apply_tfms(tfm, do_resolve=False), t)
    inputs = [[[0,4], [4,4]], [[0,0], [4,0]], [[4,0], [4,4]], [[0,0], [0,4]]]
    targets = [[[1,4], [3,4]], [[1,0], [3,0]], [[4,1], [4,3]], [[0,1], [0,3]]]
    for k, (i,t) in enumerate(zip(inputs, targets)):
        tfm = [tilt(magnitude=-0.5)]
        tfm[0].resolved = {'direction':k, 'magnitude':-0.5}
        check_image(img_test(i).apply_tfms(tfm, do_resolve=False), t)

def test_all_dihedral():
    tfm = dihedral()
    img = img_test([0,1])
    targets = [[0,1], [4,1], [0,3], [4,3], [1,0], [1,4], [3,0], [3,4]]
    for k, t in enumerate(targets):
        tfm.resolved = {'k':k}
        check_image(img.apply_tfms(tfm, do_resolve=False), t)

def test_deterministic_transforms():
    img = img_test([3,3])
    check_tfms(img, [rotate(degrees=90), rotate(degrees=-90), flip_lr(), flip_affine()],
               [[1,3], [3,1], [3,1], [3,1]])
    check_tfms(img, [zoom(scale=2), squish(scale=0.5), squish(scale=2)],
               [[4,4], [3,4], [4,3]], mode='nearest')
    crops = [crop(size=4, row_pct=r, col_pct=c) for r,c in zip([0.,0.,0.5,0.99,0.99], [0.,0.99,0.5,0.,0.99])]
    check_tfms(img, crops, [[3,3], [3,2],[2,2],[2,3],[2,2]])
    pads = [pad(padding=1, mode=mode) for mode in ['zeros', 'border', 'reflection']]
    check_tfms(img_test([3,4]), pads, [[4,5], [[4,5],[4,6]], [[4,5],[6,5]]])

def test_crop_without_size():
    path = untar_data(URLs.MNIST_TINY)/'train'/'3'
    files = get_image_files(path)
    img = open_image(path/files[0])
    tfms = get_transforms()
    img = img.apply_tfms(tfms[0])
