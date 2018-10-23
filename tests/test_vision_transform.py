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

def test_points_data_aug():
    "Check that ImagePoints get changed with their input Image."
    points = torch.randint(0,64, ((5,2)))
    img = points2pic(points, 64)
    pnts = ImagePoints(FlowField((64,64), points.float()))
    tfms = get_transforms()
    tfm_x = apply_tfms(tfms[0], img, size=64, mode='nearest')
    tfm_y = apply_tfms(tfms[0], pnts, do_resolve=False, size=64)
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
    bbox = ImageBBox.create([list(corners)], 64, 64)
    tfms = get_transforms()
    tfm_x = apply_tfms(tfms[0], img, size=64, mode='nearest', padding_mode='zeros')
    tfm_y = apply_tfms(tfms[0], bbox, do_resolve=False, size=64, padding_mode='zeros')
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
    tfms = get_transforms()
    tfm_x = apply_tfms(tfms[0], img, size=64, mode='nearest')
    tfm_y = apply_tfms(tfms[0], mask, do_resolve=False, size=64)
    new_mask = (tfm_x.data[0] > 0.5)
    assert (new_mask.float() - tfm_y.data[0].float()).sum() < 1.