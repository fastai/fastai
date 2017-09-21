from .imports import *
from .layer_optimizer import *
from enum import Enum

imagenet_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1,1,3))
def preprocess_imagenet(x): return x[..., ::-1] - imagenet_mean
def preprocess_scale(x): return ((x/255.)-0.5)*2

def scale_min(im, targ):
    r,c,_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz)

def zoom_cv(x,z):
    if z==0: return x
    rows=x.shape[0]; cols=x.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),0,z+1.)
    return cv2.warpAffine(x,M,(cols,rows))

def stretch_cv(x,sr,sc):
    if sr==0 and sc==0: return x
    r=x.shape[0]; c=x.shape[1]
    x = cv2.resize(x, None, fx=sr+1, fy=sc+1)
    nr=x.shape[0]; nc=x.shape[1]
    cr = (nr-r)//2; cc = (nc-c)//2
    return x[cr:r+cr, cc:c+cc]

def dihedral(x, dih):
    x = np.rot90(x, self.dih%4)
    return x if self.dih<4 else np.fliplr(x)

def lighting(im, b, c):
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

def rotate_cv(img, deg, mode=cv2.BORDER_REFLECT):
    rows=img.shape[0]; cols=img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)
    return cv2.warpAffine(img,M,(cols,rows), borderMode=mode)

def center_crop(im, min_sz=None):
    r,c,_ = im.shape
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
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT):
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
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return (x-self.m)/self.s

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

class ReflectionPad():
    def __init__(self, pad, mode=cv2.BORDER_REFLECT):
        self.pad,self.mode = pad,mode

    def add_pad(self, img):
        return cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad, self.mode)

    def __call__(self, x, y=None):
        x = self.add_pad(x)
        if y is not None: return x, self.add_pad(y)
        else: return x

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

TfmType = Enum('TfmType', 'NO PIXEL COORD')

class Transform():
    def __init__(tfm_y): self.tfm_y=tfm_y
    def __call__(self, x, y):
        x,y = (self.transform(x),y if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y==TfmType.PIXEL
                else self.transform_coord(x,y))


class RandomDihedralXY(Transform):
    def rand_gen(self):
        return random.randint(0,3), random.random()<0.5

    def transform_coord(self, x, y):
        rot_times, do_flip = self.rand_gen()
        x = do_transform(x, rot_times, do_flip)
        raise NotImplementedError # XXX: Handle y coord transform
        return x, y

    def transform(self, x, y=None):
        rot_times, do_flip = self.rand_gen()
        x = do_transform(x, rot_times, do_flip)
        return (x, do_transform(y, rot_times, do_flip)) if y else x

    def do_transform(self, x, rot_times, do_flip):
        x = np.rot90(x, rot_times)
        return np.fliplr(x).copy() if do_flip else x


def RandomFlip(): return lambda x: x if random.random()<0.5 else np.fliplr(x).copy()

def channel_dim(x): return np.rollaxis(x, 2)

def compose(im, fns):
    for fn in fns: im=fn(im)
    return im

class Transforms():
    def __init__(self, sz, tfms, denorm, rand_crop=False):
        self.sz,self.denorm = sz,denorm
        crop_fn = RandomCrop if rand_crop else CenterCrop
        self.tfms = tfms + [crop_fn(sz), channel_dim]
    def __call__(self, im, y): return compose(im, self.tfms), y

def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0):
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    scale = [RandomScale(sz, max_zoom) if max_zoom is not None else Scale(sz)]
    if pad: scale.append(ReflectionPad(pad))
    return Transforms(sz+pad, scale + tfms + [normalizer], denorm,
                      rand_crop=max_zoom is not None)

def noop(x): return x

transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]
