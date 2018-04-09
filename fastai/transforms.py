from .imports import *
from .layer_optimizer import *
from enum import IntEnum

def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)

def zoom_cv(x,z):
    if z==0: return x
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r))

def stretch_cv(x,sr,sc,interpolation=cv2.INTER_AREA):
    if sr==0 and sc==0: return x
    r,c,*_ = im.shape
    x = cv2.resize(x, None, fx=sr+1, fy=sc+1, interpolation=interpolation)
    nr,nc,*_ = im.shape
    cr = (nr-r)//2; cc = (nc-c)//2
    return x[cr:r+cr, cc:c+cc]

def dihedral(x, dih):
    x = np.rot90(x, dih%4)
    return x if dih<4 else np.fliplr(x)

def lighting(im, b, c):
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def rotate_cv(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def no_crop(im, min_sz=None, interpolation=cv2.INTER_AREA):
    """ Returns a squared resized image """
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    return cv2.resize(im, (min_sz, min_sz), interpolation=interpolation)

def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop(im, start_r, start_c, min_sz)

def scale_to(x, ratio, targ): return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz): return im[r:r+sz, c:c+sz]

def det_dihedral(dih): return lambda x: dihedral(x, dih)
def det_stretch(sr, sc): return lambda x: stretch_cv(x, sr, sc)
def det_lighting(b, c): return lambda x: lighting(x, b, c)
def det_rotate(deg): return lambda x: rotate_cv(x, deg)
def det_zoom(zoom): return lambda x: zoom_cv(x, zoom)

def rand0(s): return random.random()*(s*2)-s


class TfmType(IntEnum):
    """ Type of transformation.

        NO: is the default, y does not get transformed when x is transformed.
        PIXEL: when x and y are images and should be transformed in the same way.
               Example: image segmentation.
        COORD: when y are coordinate or x in which case x and y have
               to be transformed accordingly.
    """
    NO = 1
    PIXEL = 2
    COORD = 3


class Denormalize():
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image.
    """
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None:
            y = (y-self.m)/self.s
        return x,y


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


def channel_dim(x, y):
    x = np.rollaxis(x, 2)
    if isinstance(y,np.ndarray) and (len(y.shape)==3):
        y = np.rollaxis(y, 2)
    return x,y


def to_bb(YY, y):
    cols,rows = np.nonzero(YY)
    if len(cols)==0: return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


def coords2px(y, x):
    """ Transforming coordinates to pixels.

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
    Y = np.zeros((r, c))
    Y[rows, cols] = 1
    return Y


class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.
    We have 3 types of transforms:
       TfmType.NO: the target y is not transformed
       TfmType.PIXEL: assumes x and y are images of the same (cols, rows) and trasforms
           them with the same paramters.
       TfmType.COORD: assumes that y are some coordinates in the image x.

    Arguments:
        tfm_y (TfmType): type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass

    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y==TfmType.PIXEL
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

    def do_transform(self, x, is_y): raise NotImplementedError


class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r,c,*_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr, y)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y,x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)


class AddPadding(CoordTransform):
    def __init__(self, pad, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.pad,self.mode = pad,mode

    def do_transform(self, im, is_y):
        return cv2.copyMakeBorder(im, self.pad, self.pad, self.pad, self.pad, self.mode)

class CenterCrop(CoordTransform):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments:
        sz (int): size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.min_sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        return center_crop(x, self.sz_y if is_y else self.min_sz)


class RandomCrop(CoordTransform):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments:
        targ (int): target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, targ_sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz,self.sz_y = targ_sz,sz_y

    def set_state(self):
        self.store.rand_r = random.uniform(0, 1)
        self.store.rand_c = random.uniform(0, 1)

    def do_transform(self, x, is_y):
        r,c,*_ = x.shape
        sz = self.sz_y if is_y else self.targ_sz
        start_r = np.floor(self.store.rand_r*(r-sz)).astype(int)
        start_c = np.floor(self.store.rand_c*(c-sz)).astype(int)
        return crop(x, start_r, start_c, sz)


class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ (int): target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return no_crop(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return no_crop(x, self.sz,   cv2.INTER_AREA   )


class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz (int): target size to scale minimum size.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return scale_min(x, self.sz,   cv2.INTER_AREA   )


class RandomScale(CoordTransform):
    """ Scales an image so that the min size is a random number between [sz, sz*max_zoom]

    This transforms (optionally) scales x,y at with the same parameters.
    Arguments:
        sz (int): target size
        max_zoom (float): float >= 1.0
        p (float): a probability for doing the random sizing
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, sz, max_zoom, p=0.75, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.max_zoom,self.p,self.sz_y = sz,max_zoom,p,sz_y

    def set_state(self):
        self.store.mult = random.uniform(1., self.max_zoom) if random.random()<self.p else 1
        self.store.new_sz = int(self.store.mult*self.sz)
        if self.sz_y is not None: self.store.new_sz_y = int(self.store.mult*self.sz_y)

    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.store.new_sz_y, cv2.INTER_NEAREST)
        else   : return scale_min(x, self.store.new_sz,   cv2.INTER_AREA   )


def random_px_rect(y, x):
    """ Returns a 2D image of the size x with random points in a square box.

    Arguments:
        y (array): Contains the coordinates of the bounding box corners
            y = [upper_row, left_col, lower_row, right_col]
        x (array): image

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
    r,c,*_ = x.shape
    Y = np.zeros((r, c))
    Y[rows, cols] = 1
    return Y


class RandomRotate(CoordTransform):
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
        super().__init__(tfm_y)
        self.deg,self.mode,self.p = deg,mode,p

    def set_state(self):
        self.store.rdeg = rand0(self.deg)
        self.store.rp = random.random()<self.p

    def do_transform(self, x, is_y):
        if self.store.rp: x = rotate_cv(x, self.store.rdeg, mode=self.mode,
                interpolation=cv2.INTER_NEAREST if is_y else cv2.INTER_AREA)
        return x


class RandomDihedral(CoordTransform):
    def set_state(self):
        self.store.rot_times = random.randint(0,3)
        self.store.do_flip = random.random()<0.5

    def do_transform(self, x, is_y):
        x = np.rot90(x, self.store.rot_times)
        return np.fliplr(x).copy() if self.store.do_flip else x


class RandomFlip(CoordTransform):
    def set_state(self):
        self.store.do_flip = random.random()<0.5

    def do_transform(self, x, is_y):
        return np.fliplr(x).copy() if self.store.do_flip else x


class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x


class RandomBlur(Transform):
    """
    Adds a gaussian blur to the image at chance.
    Multiple blur strengths can be configured, one of them is used by random chance.
    """

    def __init__(self, blur_strengths=5, probability=0.5, tfm_y=TfmType.NO):
        # Blur strength must be an odd number, because it is used as a kernel size.
        super().__init__(tfm_y)
        self.blur_strengths = (np.array(blur_strengths, ndmin=1) * 2) - 1
        if np.any(self.blur_strengths < 0):
            raise ValueError("all blur_strengths must be > 0")
        self.probability = probability
        self.apply_transform = False

    def set_state(self):
        self.store.apply_transform = random.random() < self.probability
        kernel_size = np.random.choice(self.blur_strengths)
        self.store.kernel = (kernel_size, kernel_size)

    def do_transform(self, x, is_y):
        return cv2.GaussianBlur(src=x, ksize=self.store.kernel, sigmaX=0) if self.apply_transform else x


def compose(im, y, fns):
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)


class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3

crop_fn_lu = {CropType.RANDOM: RandomCrop, CropType.CENTER: CenterCrop, CropType.NO: NoCrop}

class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER, tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms + [crop_tfm, normalizer, channel_dim]
    def __call__(self, im, y=None): return compose(im, y, self.tfms)


def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
             else Scale(sz, tfm_y, sz_y=sz_y)]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    #if (max_zoom is not None or pad!=0) and crop_type is None: crop_type = CropType.RANDOM
    return Transforms(sz, scale + tfms, normalizer, denorm, crop_type, tfm_y=tfm_y, sz_y=sz_y)

def noop(x): return x

transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
inception_stats = A([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)

def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    if aug_tfms is None: aug_tfms=[]
    tfm_norm = Normalize(*stats, tfm_y=tfm_y)
    tfm_denorm = Denormalize(*stats)
    val_crop = CropType.CENTER if crop_type==CropType.RANDOM else crop_type
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop, tfm_y=tfm_y, sz_y=sz_y)
    trn_tfm=image_gen(tfm_norm, tfm_denorm, sz, tfms=aug_tfms, max_zoom=max_zoom,
                      pad=pad, crop_type=crop_type, tfm_y=tfm_y, sz_y=sz_y, pad_mode=pad_mode)
    return trn_tfm, val_tfm


def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    stats = inception_stats if f_model in inception_models else imagenet_stats
    return tfms_from_stats(stats, sz, aug_tfms, max_zoom=max_zoom, pad=pad, crop_type=crop_type,
                       tfm_y=tfm_y, sz_y=sz_y, pad_mode=pad_mode)

