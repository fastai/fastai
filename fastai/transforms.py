from .imports import *
from .layer_optimizer import *
from enum import IntEnum

def scale_min(im, sz, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        sz (int): target size
    """
    r,c,*_ = im.shape
    ratio = sz/min(r,c)
    dsize = (scale_to(c, ratio, sz), scale_to(r, ratio, sz))
    return cv2.resize(im, dsize, interpolation=interpolation)

def zoom_cv(im, zoom):
    '''zooms the center of image x, by a factor of z+1 while retaining the origal image size and proportion. '''
    if zoom==0: return im
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,zoom+1.)
    return cv2.warpAffine(im,M,(c,r))

def stretch_cv(x,sr,sc,interpolation=cv2.INTER_AREA):
    '''stretches image x horizontally by sr+1, and vertically by sc+1 while retaining the origal image size and proportion.'''
    if sr==0 and sc==0: return x
    r,c,*_ = x.shape
    x = cv2.resize(x, None, fx=sr+1, fy=sc+1, interpolation=interpolation)
    nr,nc,*_ = x.shape
    cr = (nr-r)//2; cc = (nc-c)//2
    return x[cr:r+cr, cc:c+cc]

def dihedral(x, dih):
    '''performs any of 8 90 rotations or flips for image x.
    '''
    x = np.rot90(x, dih%4)
    return x if dih<4 else np.fliplr(x)

def lighting(im, b, c):
    ''' adjusts image's balance and contrast'''
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

def no_crop(im, sz=None, interpolation=cv2.INTER_AREA):
    """ Returns a squared resized image """
    r,c,*_ = im.shape
    if sz is None: sz = min(r,c)
    return cv2.resize(im, (sz, sz), interpolation=interpolation)

def center_crop(im, sz=None):
    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if sz is None: sz = min(r, c)
    start_r = math.ceil((r - sz) / 2)
    start_c = math.ceil((c - sz) / 2)
    return crop(im, start_r, start_c, sz)

def googlenet_resize(im, sz, min_area_frac, min_aspect_ratio, max_aspect_ratio, flip_hw_p, interpolation=cv2.INTER_AREA):
    """ Randomly crops an image with an aspect ratio and returns a squared resized image of size targ
    
    References:
    1. https://arxiv.org/pdf/1409.4842.pdf
    2. https://arxiv.org/pdf/1802.07888.pdf
    """
    h,w,*_ = im.shape
    area = h*w
    for _ in range(10):
        targetArea = random.uniform(min_area_frac, 1.0) * area
        aspectR = random.uniform(min_aspect_ratio, max_aspect_ratio)
        ww = int(np.sqrt(targetArea * aspectR) + 0.5)
        hh = int(np.sqrt(targetArea / aspectR) + 0.5)
        if flip_hw_p:
            ww, hh = hh, ww
        if hh <= h and ww <= w:
            x1 = 0 if w == ww else random.randint(0, w - ww)
            y1 = 0 if h == hh else random.randint(0, h - hh)
            out = im[y1:y1 + hh, x1:x1 + ww]
            out = cv2.resize(out, (sz, sz), interpolation=interpolation)
            return out
    out = scale_min(im, sz, interpolation=interpolation)
    out = center_crop(out)
    return out

def cutout(im, n_holes, length):
    ''' cuts out n_holes number of square holes of size length in image at random locations. holes may be overlapping. '''
    r,c,*_ = im.shape
    mask = np.ones((r, c), np.int32)
    for n in range(n_holes):
        y = np.random.randint(r)
        x = np.random.randint(c)

        y1 = int(np.clip(y - length / 2, 0, r))
        y2 = int(np.clip(y + length / 2, 0, r))
        x1 = int(np.clip(x - length / 2, 0, c))
        x2 = int(np.clip(x + length / 2, 0, c))
        mask[y1: y2, x1: x2] = 0.
    
    mask = mask[:,:,None]
    im = im * mask
    return im

def scale_to(x, ratio, targ): 
    '''Calculate dimension of an image during scaling with aspect ratio'''
    return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz): 
    '''
    crop image into a square of size sz, 
    '''
    return im[r:r+sz, c:c+sz]

def det_dihedral(dih): return lambda x: dihedral(x, dih)
def det_stretch(sr, sc): return lambda x: stretch_cv(x, sr, sc)
def det_lighting(b, c): return lambda x: lighting(x, b, c)
def det_rotate(deg): return lambda x: rotate_cv(x, deg)
def det_zoom(zoom): return lambda x: zoom_cv(x, zoom)

def rand0(s): return random.random()*(s*2)-s

def make_rect(coords, x_shape):
    r, c, *_ = x_shape
    im = np.zeros((r, c))
    r1, c1, r2, c2 = coords.astype(np.int)
    im[r1:r2, c1:c2] = 1.
    return im

def transform_coords(t, coords, im_shape, **kwargs):
    coords_part = partition(coords, 4)
    new_coords = [to_bb(t(make_rect(c, im_shape, **kwargs), **TfmParams.CLASS))
                  for c in coords_part]
    return np.concatenate(new_coords)

class TfmType(object):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4

class DetTfm():
    def __init__(self, func=None,  **kwargs):
        if func is None:
            func = self.do
        self._tfm_y = TfmType.NO  # deprecated
        if 'tfm_y' in kwargs:
            warnings.warn("tfm_y is deprecated", DeprecationWarning)
            self.tfm_y = kwargs['tfm_y'] # to show it in __repr__
            self._tfm_y = kwargs['tfm_y']
        self.func = partial(func, **kwargs)

    do = lambda x, **kwargs: x
    def determ(self): return self
    def partial(self, **kwargs): return DetTfm(self.func, **kwargs)
    def __call__(self, im, y=DeprecationWarning, **kwargs):
        if y is not DeprecationWarning:
            warnings.warn("__call__(x,y) is deprecated, use __call__(x) and __call__(y, **TfmParams.CLASS)",
                          DeprecationWarning)
            return self._depr_call(im, y)
        return self._new_call(im, **kwargs)

    def _new_call(self, im, **kwargs):
        return self.func(im, **kwargs)

    def _depr_call(self, x, y):
        t = self.determ()
        if self._tfm_y is TfmType.NO:
            return (t(x),y)
        if self._tfm_y is (TfmType.COORD):
            return (t(x), transform_coords(t, y, x.shape))
        if self._tfm_y is TfmType.PIXEL or self._tfm_y is TfmType.CLASS:
            return (t(x),t(y, **self._tfm_y))
        raise NotImplementedError(f"Unkonwn tfm_y: {self._tfm_y} type")

    def __str__(self):
        applied = ['_']+[f"{n}={self.func.keywords.get(n, '?')}" for n in self.func.func.__code__.co_varnames
                        if n != 'kwargs' and not n == 'im']
        fn = self.func.func.__qualname__
        return f"{fn}({', '.join(applied)})"

class RndTfm(DetTfm):
    """ Representation of a transformation
    """

    def new_state(self): return {}

    def determ(self): return self.partial(
        **self.new_state())  # Todo get rid of new state

    def _new_call(self, im, **kwargs):
        return self.determ()(im)

class Denormalize(DetTfm):
    """ De-normalizes an image, returning it to original format."""

    @staticmethod
    def do(im, mean, stddev, **kwargs): return im * stddev + mean

    def __init__(self, m, s, **kwargs):
        super().__init__(mean=np.array(m, dtype=np.float32),
                         stddev=np.array(s, dtype=np.float32), **kwargs)

class Normalize(DetTfm):
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s, **kwargs):
        super().__init__(mean=np.array(m, dtype=np.float32),
                         stddev=np.array(s, dtype=np.float32), **kwargs)

    @staticmethod
    def do(im, mean, stddev, **kwargs): return (im - mean) / stddev

class ChannelOrder(DetTfm):
    '''
    changes image array shape from (h, w, 3) to (3, h, w).
    '''

    @staticmethod
    def do(im, **kwargs):
        if len(im.shape) == 2:
            return im  #[None, ...]  # mask ## TODO: coord requires im to stay 2d
        return np.rollaxis(im, 2)

def to_bb(YY, y="deprecated"):
    """Convert mask YY to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(YY)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def coords2px(y, x):
    """ Transforming coordinates to pixels.

    Arguments:
        y : np array
            vector in which (y[0], y[1]) and (y[2], y[3]) are the
            the corners of a bounding box.
        x : image
            an image
    Returns:
        Y : image
            of shape x.shape
    """
    rows = np.rint([y[0], y[0], y[2], y[2]]).astype(int)
    cols = np.rint([y[1], y[3], y[1], y[3]]).astype(int)
    r,c,*_ = x.shape
    Y = np.zeros((r, c))
    Y[rows, cols] = 1
    return Y

class Transform(RndTfm):  # to keep the diff meaningful
    """ A class that represents a transform.
    All other transforms should subclass it. All subclasses should override
    do_transform.
    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    pass

class CoordTransform(RndTfm): # to keep the diff meaningful\
    """ A coordinate transform.  """
    pass

class AddPadding(CoordTransform):
    """ A class that represents adding paddings to an image.

    The default padding is border_reflect
    Arguments
    ---------
        pad : int
            size of padding on top, bottom, left and right
        mode:
            type of cv2 padding modes. (e.g., constant, reflect, wrap, replicate. etc. )
    """
    def __init__(self, pad, mode=cv2.BORDER_REFLECT, **kwargs):
        super().__init__(pad=pad, pad_mode=mode, **kwargs)

    @staticmethod
    def do(im, pad, pad_mode, **kwargs):
        return cv2.copyMakeBorder(im, pad, pad, pad, pad, pad_mode)

class CenterCrop(CoordTransform):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        sz: int
            size of the crop.
    """
    def __init__(self, sz, **kwargs):
        super().__init__(sz=sz, **kwargs)

    @staticmethod
    def do(im, sz=None, **kwargs): return center_crop(im,sz=sz)

class RandomCrop(CoordTransform):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        targ: int
            target size of the crop.
    """
    def __init__(self, sz, **kwargs):
        super().__init__(sz=sz, **kwargs)

    def determ(self):
        return self.partial(_rand_r=random.uniform(0, 1), _rand_c=random.uniform(0, 1))

    @staticmethod
    def do(im, sz=None, _rand_r=None, _rand_c=None, **kwargs):
        r, c, *_ = im.shape
        start_r = np.floor(_rand_r * (r - sz)).astype(int)
        start_c = np.floor(_rand_c * (c - sz)).astype(int)
        return crop(im, start_r, start_c, sz)

class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ: int
            target size of the crop.
    """
    def __init__(self, sz, **kwargs):
        super().__init__(sz=sz, interpolation=cv2.INTER_AREA, **kwargs)

    do = staticmethod(no_crop)
    @staticmethod

    def do(im, sz=None, interpolation=cv2.INTER_AREA, **kwargs):
        return no_crop(im,sz,interpolation=interpolation)

class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
    """
    def __init__(self, sz, **kwargs):
        super().__init__(sz=sz, interpolation=cv2.INTER_AREA, **kwargs)

    @staticmethod
    def do(im, sz, interpolation, **kwargs): return scale_min(im, sz, interpolation)

class RandomScale(CoordTransform):
    """ Scales an image so that the min size is a random number between [sz, sz*max_zoom]

    This transforms (optionally) scales x,y at with the same parameters.
    Arguments:
        sz: int
            target size
        max_zoom: float
            float >= 1.0
        p : float
            a probability for doing the random sizing
    """

    def __init__(self, sz, max_zoom, p=0.75, **kwargs):
        self.max_zoom,self.p = max_zoom,p
        super().__init__(sz=sz, interpolation=cv2.INTER_AREA, **kwargs)

    def determ(self):
        min_z = 1.
        max_z = self.max_zoom
        if isinstance(self.max_zoom, collections.Iterable):
            min_z, max_z = self.max_zoom
        zoom = random.uniform(min_z, max_z) if random.random()<self.p else 1
        return self.partial(zoom=zoom)

    @staticmethod
    def do(im, sz, interpolation, zoom):
        return scale_min(im, int(sz*zoom), interpolation=interpolation)

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
    def __init__(self, deg, p=0.75, pad_mode=cv2.BORDER_REFLECT, **kwargs):
        super().__init__(pad_mode=pad_mode, interpolation=cv2.INTER_AREA, **kwargs)
        self.deg,self.p = deg,p

    def determ(self):
        return self.partial(rdeg=rand0(self.deg),
                            rp=random.random()<self.p)

    @staticmethod
    def do(im, pad_mode, interpolation, rp, rdeg, **kwargs):
        if rp:
            im = rotate_cv(im, rdeg, pad_mode, interpolation)
        return im

class RandomDihedral(CoordTransform):
    """
    Rotates images by random multiples of 90 degrees and/or reflection.
    Please reference D8(dihedral group of order eight), the group of all symmetries of the square.
    """
    def new_state(self):
        return dict(rot_times=random.randint(0,3), do_flip=random.random()<0.5)

    @staticmethod
    def do(im, rot_times, do_flip, **kwargs):
        im = np.rot90(im, rot_times)  # FIXME: use dihedral?
        if do_flip:
            im = np.fliplr(im).copy()
        return im

class RandomFlip(CoordTransform):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p=p
    def new_state(self): return dict(do_flip=random.random()<self.p)
    @staticmethod
    def do(im, do_flip, **kwargs):
        if do_flip:
            return np.fliplr(im).copy()
        return im

class RandomLighting(Transform):
    def __init__(self, b, c, **kwargs):
        super().__init__(**kwargs)
        self.b,self.c = b,c

    def new_state(self):
        return dict(b=rand0(self.b), c=rand0(self.c))

    @staticmethod
    def do(im, b, c, **kwargs):
        c = -1/(c-1) if c<0 else c+1
        im = lighting(im, b, c)
        return im

class RandomRotateZoom(CoordTransform):
    """
        Selects between a rotate, zoom, stretch, or no transform.
        Arguments:
            deg - maximum degrees of rotation.
            zoom - maximum fraction of zoom.
            stretch - maximum fraction of stretch.
            ps - probabilities for each transform. List of length 4. The order for these probabilities is as listed respectively (4th probability is 'no transform').
    """
    def __init__(self, deg, zoom, stretch, ps=None, pad_mode=cv2.BORDER_REFLECT,**kwargs):
        super().__init__(**kwargs)
        if ps is None: ps = [0.25,0.25,0.25,0.25]
        assert len(ps) == 4, 'does not have 4 probabilities for p, it has %d' % len(ps)
        self.tfms = RandomRotate(deg, p=1, pad_mode=pad_mode), RandomZoom(zoom), RandomStretch(stretch)
        self.pass_t = PassThru()
        self.cum_ps = np.cumsum(ps)
        assert self.cum_ps[3]==1, 'probabilites do not sum to 1; they sum to %d' % self.cum_ps[3]

    def determ(self):
        tfm = self.pass_t
        choice = self.cum_ps[3]*random.random()
        for i in range(len(self.tfms)):
            if choice < self.cum_ps[i]:
                tfm = self.tfms[i]
                break
        return tfm.determ()

class RandomZoom(CoordTransform):

    def __init__(self, zoom_max, zoom_min=0, pad_mode=cv2.BORDER_REFLECT,**kwargs):
        super().__init__()
        self.zoom_max, self.zoom_min = zoom_max, zoom_min

    def new_state(self):
        return dict(zoom=self.zoom_min+(self.zoom_max-self.zoom_min)*random.random())

    @staticmethod
    def do(im, zoom, **kwargs): return zoom_cv(im, zoom)

class RandomStretch(CoordTransform):
    def __init__(self, max_stretch, ):
        super().__init__()
        self.max_stretch = max_stretch

    def new_state(self):
        return dict(stretch=self.max_stretch*random.random(), stretch_dir=random.randint(0,1))

    @staticmethod
    def do(im, stretch, stretch_dir, **kwargs):
        return stretch_cv(im, stretch*(stretch_dir), stretch*(1-stretch_dir))

class PassThru(Transform):
    def ttm(self, im, **kwargs): return im

class RandomBlur(Transform):
    """
    Adds a gaussian blur to the image at chance.
    Multiple blur strengths can be configured, one of them is used by random chance.
    """
    def __init__(self, blur_strengths=5, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        # Blur strength must be an odd number, because it is used as a kernel size.
        self.blur_strengths = (np.array(blur_strengths, ndmin=1) * 2) - 1
        if np.any(self.blur_strengths < 0):
            raise ValueError("all blur_strengths must be > 0")
        self.probability = probability

    def new_state(self):
        kernel_size = np.random.choice(self.blur_strengths)
        return dict(apply_transform = random.random() < self.probability, kernel = (kernel_size, kernel_size))

    @staticmethod
    def do(im, apply_transform, kernel, **kwargs):
        return cv2.GaussianBlur(src=im, ksize=kernel, sigmaX=0) if apply_transform else im

class Cutout(Transform):
    def __init__(self, n_holes, length, **kwargs):
        super().__init__(cutout, n_holes=n_holes, length=length, **kwargs)

class GoogleNetResize(CoordTransform):
    """ Randomly crops an image with an aspect ratio and returns a squared resized image of size targ

    Arguments:
        targ_sz: int
            target size
        min_area_frac: float < 1.0
            minimum area of the original image for cropping
        min_aspect_ratio : float
            minimum aspect ratio
        max_aspect_ratio : float
            maximum aspect ratio
        flip_hw_p : float
            probability for flipping magnitudes of height and width
        tfm_y: TfmType
            type of y transform
    """

    def __init__(self, targ_sz, min_area_frac=0.08, min_aspect_ratio=0.75, max_aspect_ratio=1.333, flip_hw_p=0.5, **kwargs):
        super().__init__(targ_sz=targ_sz, min_area_frac=min_area_frac, min_aspect_ratio=min_aspect_ratio,
                        max_aspect_ratio=max_aspect_ratio, interpolation=cv2.INTER_AREA, **kwargs)
        self.flip_hw_p = flip_hw_p

    def new_state(self):
        return dict(fp=random.random()<self.flip_hw_p)

    @staticmethod
    def do(im, sz, min_area_frac, min_aspect_ratio, max_aspect_ratio, flip_hw_p, interpolation=cv2.INTER_AREA):
        return googlenet_resize(im,
                                sz=sz,
                                min_area_frac=min_area_frac,
                                min_aspect_ratio=min_aspect_ratio,
                                max_aspect_ratio=max_aspect_ratio,
                                flip_hw_p=flip_hw_p,
                                interpolation=interpolation)

class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4

class TfmParams(object):
    CLASS = {
        # We use string names to make sure that we won't get errors during module reloading
        'disable': (RandomLighting.do.__qualname__,
                    RandomDihedral.do.__qualname__,
                    RandomBlur.do.__qualname__,
                    Normalize.do.__qualname__,
                    Denormalize.do.__qualname__),
        'interpolation': cv2.INTER_NEAREST,
        'pad_mode': cv2.BORDER_CONSTANT,
    }
    @staticmethod
    def from_type(tfm):
        if tfm == TfmType.CLASS or tfm == TfmType.COORD:
            return TfmParams.CLASS
        return {}

class ComposedTransform():
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, im, **kwargs):
        return self.do_all(im, **kwargs)

    @staticmethod
    def tfm_func_name(t):
        if hasattr(t, 'func'): return ComposedTransform.tfm_func_name(t.func)
        return t.__qualname__

    def do_all(self, im, disable=(),  **kwargs):
        for t in self.tfms:
            if self.tfm_func_name(t) not in disable:
                im = t(im, **kwargs)
        return im

    def determ(self): return ComposedTransform([t.determ() for t in self.tfms])

    def __repr__(self): return f"%s(tfms=[\n  %s\n])" % (self.__class__.__name__, ",\n  ".join(map(str,self.tfms)))

crop_fn_lu = {CropType.RANDOM: RandomCrop, CropType.CENTER: CenterCrop, CropType.NO: NoCrop, CropType.GOOGLENET: GoogleNetResize}

class Transforms(ComposedTransform):
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz)
        tfms.append(crop_tfm)
        if normalizer is not None: tfms.append(normalizer)
        tfms.append(ChannelOrder())
        self.tfm = ComposedTransform(tfms)
        self.tfm_y = tfm_y

    def determ(self):
        return self.tfm.determ()

    def __call__(self, im, y=None):
        t = self.determ()
        return self.apply_transforms(t, im, y)

    def apply_transforms(self, t, x, y): # for compatibility
        nx = t(x)
        if y is None: return nx
        ny = y
        if self.sz_y is not None: t = partial(t, sz=self.sz_y)
        if self.tfm_y is not TfmType.NO:
            ny = transform_coords(t, y, x.shape) if self.tfm_y is TfmType.COORD else t(y, **TfmParams.from_type(self.tfm_y))
        return nx,ny

    def __repr__(self): return "Trasnforms(...)\n# with "+ repr(self.tfm)

def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, scale=None):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    if scale is None:
        scale = [RandomScale(sz, max_zoom) if max_zoom is not None
                 else Scale(sz)]
    elif not is_listy(scale): scale = [scale]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

def noop(x):
    """dummy function for do-nothing.
    equivalent to: lambda x: x"""
    return x

transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
"""Statistics pertaining to image data from image net. mean and std of the images of each color channel"""
inception_stats = A([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)

def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    if aug_tfms is None: aug_tfms=[]
    tfm_norm = Normalize(*stats, tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    val_crop = CropType.CENTER if crop_type in (CropType.RANDOM,CropType.GOOGLENET) else crop_type
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop,
            tfm_y=tfm_y, sz_y=sz_y, scale=scale)
    trn_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type,
            tfm_y=tfm_y, sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, scale=scale)
    return trn_tfm, val_tfm


def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Returns separate transformers of images for training and validation.
    Transformers are constructed according to the image statistics given b y the model. (See tfms_from_stats)

    Arguments:
        f_model: model, pretrained or not pretrained
    """
    stats = inception_stats if f_model in inception_models else imagenet_stats
    return tfms_from_stats(stats, sz, aug_tfms, max_zoom=max_zoom, pad=pad, crop_type=crop_type,
                           tfm_y=tfm_y, sz_y=sz_y, pad_mode=pad_mode, norm_y=norm_y, scale=scale)

