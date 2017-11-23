from .imports import *
from .layer_optimizer import *
from enum import IntEnum

def scale_min(im, targ):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz)

def zoom_cv(x,z):
    if z==0: return x
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r))

def stretch_cv(x,sr,sc):
    if sr==0 and sc==0: return x
    r,c,*_ = im.shape
    x = cv2.resize(x, None, fx=sr+1, fy=sc+1)
    nr,nc,*_ = im.shape
    cr = (nr-r)//2; cc = (nc-c)//2
    return x[cr:r+cr, cc:c+cc]

def dihedral(x, dih):
    x = np.rot90(x, self.dih%4)
    return x if self.dih<4 else np.fliplr(x)

def lighting(im, b, c):
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=flags)

def no_crop(im, min_sz=None):
    """ Returns a squared resized image """
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    return cv2.resize(im, (min_sz, min_sz))

def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop(im, start_r, start_c, min_sz)

def scale_to(x, ratio, targ): return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz): return im[r:r+sz, c:c+sz]

def det_dihedral(dih): return lambda x: dihedral(dih)
def det_stretch(sr, sc): return lambda x: stretch_cv(x, sr, sc)
def det_lighting(b, c): return lambda x: lighting(x, b, c)
def det_rotate(deg): return lambda x: rotate_cv(x, deg)
def det_zoom(zoom): return lambda x: zoom_cv(x, zoom)

def rand0(s): return random.random()*(s*2)-s

def CenterCrop(min_sz=None): return lambda x: center_crop(x, min_sz)
def NoCrop(min_sz=None): return lambda x: no_crop(x, min_sz)
def Scale(sz): return lambda x: scale_min(x, sz)


class RandomScale():
    def __init__(self, targ, max_zoom, p=0.75):
        self.targ,self.max_zoom,self.p = targ,max_zoom,p
    def __call__(self, x):
        if random.random()<self.p:
            sz = int(random.uniform(1., self.max_zoom)*self.targ)
        else: sz = self.targ
        return scale_min(x, sz)


class RandomRotate():
    def __init__(self, deg, mode=cv2.BORDER_REFLECT):
        self.deg,self.mode,self.p = deg,mode,p
    def __call__(self, x, y=None):
        deg = rand0(self.deg)
        if random.random()<self.p:
            x = rotate_cv(x, deg, self.mode)
            if y is not None: y = rotate_cv(y, deg, self.mode)
        return x if y is None else (x,y)


class RandomCrop():
    def __init__(self, targ): self.targ = targ
    def __call__(self, x):
        r,c,_ = x.shape
        start_r = random.randint(0, r-self.targ)
        start_c = random.randint(0, c-self.targ)
        res = crop(x, start_r, start_c, self.targ)
        return res


class Denormalize():
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x, y): return (x-self.m)/self.s, y


class RandomRotateZoom():
    def __init__(self, deg, zoom, stretch, mode=cv2.BORDER_REFLECT):
        self.deg,self.zoom,self.stretch,self.mode = deg,zoom,stretch,mode

    def __call__(self, x, y=None):
        choice = random.randint(0,3)
        if choice==0: pass
        elif choice==1: x = rotate_cv(x, rand0(self.deg), self.mode)
        elif choice==2: x = zoom_cv(x, random.random()*self.zoom)
        elif choice==3:
            str_choice = random.randint(0,1)
            sa = random.random()*self.stretch
            if str_choice==0: x = stretch_cv(x, sa, 0)
            else:             x = stretch_cv(x, 0, sa)
        assert (y is None) # not implemented
        return x


class RandomLighting():
    def __init__(self, b, c): self.b,self.c = b,c

    def __call__(self, x, y=None):
        b = rand0(self.b)
        c = rand0(self.c)
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        if y is not None: return x, lighting(y, b, c)
        else: return x


class RandomDihedral():
    def __call__(self, x):
        x = np.rot90(x, random.randint(0,3))
        return x.copy() if random.random()<0.5 else np.fliplr(x).copy()

def RandomFlip(): return lambda x: x if random.random()<0.5 else np.fliplr(x).copy()

def channel_dim(x, y): return np.rollaxis(x, 2), y

def to_bb(YY, y):
    (cols, rows) = np.nonzero(YY)
    # return 0s when the fish has been cropped
    if rows.shape[0] == 0:
        #TODO: log this somewhere
        print("loss my fish")
        return np.zeros(4)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row])

def coords2px(y, x):
    """ Transforming coordinates to pixes.

    Arguments:
        y (np array): vector in which (y[0], y[1]) and (y[2], y[3]) are the
            the corners of a bounding box.
        x (image): an image
    Returns:
        Y (image): of shape x.shape
    """
    rows = np.rint([y[0], y[0], y[2], y[2]]).astype(int)
    cols = np.rint([y[1], y[3], y[1], y[3]]).astype(int)
    r,c,*_ = x.shape
    Y = np.zeros((c, r))
    Y[rows, cols] = 1
    return Y


class TfmType(IntEnum):
    """ Type of transformation.

        NO: is the default, y does not get transformed when x is transformed.
        PIXEL: when x and y are images and should be transformed in the same way.
               Example: image segmentation.
        COORD: when y are coordinate or x in which case x and y have
               to be transformed accordingly. Example: bounding box for fish dataset.
    """
    NO = 1
    PIXEL = 2
    COORD = 3


class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.
    We have 3 types of transforms:
       TfmType.NO: the target y is not transformed
       TfmType.PIXEL: assumes x and y are images of the same (cols, rows) and trasforms
           them with the same paramters.
       TfmType.COORD: assumes that y are some coordinates in the image x. At the momemnt this
           works for a bounding box around a fish.

    Arguments:
        tfm_y (TfmType): type of transform
    """
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y
    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y==TfmType.PIXEL
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x)
        return (x, self.do_transform(y)) if y is not None else x

    def do_transform(self, x): raise NotImplementedError


class CoordTransform(Transform):
    """ A class that represents a coordinate transform.

    Note: at the moment this works for the bounding box problem.
    """
    def transform_coord(self, x, y):
        y = coords2px(y, x)
        x,y_tr = self.transform(x,y)
        return x, to_bb(y_tr, y)


class AddPadding(CoordTransform):
    def __init__(self, pad, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        self.pad,self.mode,self.tfm_y = pad,mode,tfm_y
    def do_transform(self, im):
        return cv2.copyMakeBorder(im, self.pad, self.pad, self.pad, self.pad, self.mode)

class CenterCropXY(CoordTransform):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments:
        sz (int): size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO): self.tfm_y,self.min_sz = tfm_y,sz
    def do_transform(self, x): return center_crop(x, self.min_sz)


class RandomCropXY(CoordTransform):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments:
        targ (int): target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, targ, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.targ=targ

    def set_state(self):
        self.rand_r = random.uniform(0, 1)
        self.rand_c = random.uniform(0, 1)

    def do_transform(self, x):
        r,c,_ = x.shape
        start_r = np.floor(self.rand_r*(r-self.targ)).astype(int)
        start_c = np.floor(self.rand_c*(c-self.targ)).astype(int)
        return crop(x, start_r, start_c, self.targ)


class NoCropXY(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ (int): target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.sz=sz

    def do_transform(self, x):
       return no_crop(x, self.sz)


class ScaleXY(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz (int): target size to scale minimum size.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.sz=sz

    def do_transform(self, x):
        return scale_min(x, self.sz)


class RandomScaleXY(CoordTransform):
    """ Scales an image so that the min size is a random number between [sz, sz*max_zoom]

    This transforms (optionally) scales x,y at with the same parameters.
    Arguments:
        sz (int): target size
        max_zoom (float): float >= 1.0
        p (float): a probability for doing the random sizing
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, sz, max_zoom, p=0.75, tfm_y=TfmType.NO):
        self.sz,self.max_zoom,self.p,self.tfm_y = sz,max_zoom,p,tfm_y

    def set_state(self):
        self.new_sz = self.sz
        if random.random()<self.p:
            self.new_sz = int(random.uniform(1., self.max_zoom)*self.sz)

    def do_transform(self, x):
        return scale_min(x, self.new_sz)


def random_px_rect(y, x):
    """ Returns a 2D image of the size x with random points in a square box.

    Arguments:
        y (array): defines a bounding box (arround a fish) for the
            fishery datset. Contains the coordinates of the bounding box corners
            y = [upper_row, left_col, lower_row, right_col]
        x (array): image (with the target fish)

    Returns:
        Y (array): A 2D array of size (x.shape[0], x.shape[1]) with pixes
            on corners of the bounding box and random points in the boundary of the box.
    """
    rows0 = np.array([y[0], y[0], y[2], y[2]])
    cols0 = np.array([y[1], y[3], y[1], y[3]])
    n = [np.random.randint(10, 20) for i in range(4)]
    rand_rows = np.hstack([np.random.uniform(y[0], y[2], size=n[i]) for i in range(2)])
    fixed_cols = np.hstack([ y[j] * np.ones(n[i]) for i, j in zip(range(0,2), [1,3])])
    rand_cols = np.hstack([np.random.uniform(y[1], y[3], size=n[i]) for i in range(2,4)])
    fixed_rows = np.hstack([y[j] * np.ones(n[i]) for i, j in zip(range(2,4),[0,2])])
    rows = np.hstack([rows0, rand_rows, fixed_rows]).astype(int)
    cols = np.hstack([cols0, fixed_cols, rand_cols]).astype(int)
    r,c = x.size
    Y = np.zeros((c, r))
    Y[rows, cols] = 1
    return Y

class RandomRotateXY(Transform):
    """ Rotates images and (optionally) target y.

    Rotating coordinates is treated differently for x and y on this
    transform.
     Arguments:
        deg (float): degree to rotate.
        p (float): probability of rotation
        mode: type of border
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        self.deg,self.mode,self.p,self.tfm_y = deg,mode,p,tfm_y

    def set_state(self):
        self.rdeg = rand0(self.deg)
        self.rp = random.random()<self.p

    def transform_coord(self, x, y):
        y = random_px_rect(y, x)
        x,y_tr = self.do_transform(x), self.do_transform_y(y)
        y = to_bb(y_tr, y)
        return x, y

    def do_transform(self, x):
        if self.rp: x = rotate_cv(x, self.rdeg, mode=self.mode)
        return x

    def do_transform_y(self, y):
        if self.rp: y = rotate_cv(y, self.rdeg, flags=cv2.INTER_LINEAR)
        return y


class RandomDihedralXY(CoordTransform):
    def set_state(self):
        self.rot_times = random.randint(0,3)
        self.do_flip = random.random()<0.5

    def do_transform(self, x):
        x = np.rot90(x, self.rot_times)
        return np.fliplr(x).copy() if self.do_flip else x


class RandomFlipXY(CoordTransform):
    def set_state(self):
        self.do_flip = random.random()<0.5

    def do_transform(self, x):
        return np.fliplr(x).copy() if self.do_flip else x


class RandomLightingXY(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.b,self.c = b,c

    def set_state(self):
        self.b_rand = rand0(self.b)
        self.c_rand = rand0(self.c)

    def do_transform(self, x):
        b = self.b
        c = self.c
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x

def compose(im, y, fns):
    for fn in fns:
        im, y =fn(im, y)
    return im if y is None else (im, y)


class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3


class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER, tfm_y=TfmType.NO):
        self.sz,self.denorm = sz,denorm
        crop_tfm = CenterCropXY(sz, tfm_y)
        if crop_type == CropType.RANDOM: crop_tfm = RandomCropXY(sz, tfm_y)
        if crop_type == CropType.NO: crop_tfm = NoCropXY(sz, tfm_y)
        self.tfms = tfms + [crop_tfm, normalizer, channel_dim]
    def __call__(self, im, y=None): return compose(im, y, self.tfms)


def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None):
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    scale = [RandomScaleXY(sz, max_zoom, tfm_y) if max_zoom is not None else ScaleXY(sz, tfm_y)]
    if pad: scale.append(AddPadding(pad))
    if (max_zoom is not None or pad!=0) and crop_type is None: crop_type = CropType.RANDOM
    return Transforms(sz, scale + tfms, normalizer, denorm, crop_type, tfm_y)

def noop(x): return x

# TODO: find a different solution now that we have tfm_y
transforms_basic    = [RandomRotateXY(10), RandomLightingXY(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlipXY()]
transforms_top_down = transforms_basic + [RandomDihedralXY()]

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
inception_stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)

def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None):
    if aug_tfms is None: aug_tfms=[]
    tfm_norm = Normalize(*stats)
    tfm_denorm = Denormalize(*stats)
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=CropType.CENTER, tfm_y=tfm_y)
    trn_tfm=image_gen(tfm_norm, tfm_denorm, sz, tfms=aug_tfms, max_zoom=max_zoom,
                      pad=pad, crop_type=crop_type, tfm_y=tfm_y)
    return trn_tfm, val_tfm

def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None):
    stats = inception_stats if f_model in inception_models else imagenet_stats
    return tfms_from_stats(stats, sz, aug_tfms, max_zoom=max_zoom, pad=pad, crop_type=crop_type, tfm_y=tfm_y)

