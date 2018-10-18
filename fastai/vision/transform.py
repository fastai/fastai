"Image transformations for data augmentation. All transforms are done on the tensor level"
from ..torch_core import *
from .image import *
from .image import _affine_mult

__all__ = ['brightness', 'contrast', 'crop', 'crop_pad', 'dihedral', 'dihedral_affine', 'flip_affine', 'flip_lr', 
           'get_transforms', 'jitter', 'pad', 'perspective_warp', 'rand_pad', 'rand_crop', 'rand_zoom', 'rotate', 'skew', 'squish',
           'rand_resize_crop', 'symmetric_warp', 'tilt', 'zoom', 'zoom_crop']

_pad_mode_convert = {'reflection':'reflect', 'zeros':'constant', 'border':'replicate'}

@TfmLighting
def brightness(x, change:uniform):
    "Apply `change` in brightness of image `x`."
    return x.add_(scipy.special.logit(change))

@TfmLighting
def contrast(x, scale:log_uniform):
    "Apply `scale` to contrast of image `x`."
    return x.mul_(scale)

@TfmAffine
def rotate(degrees:uniform):
    "Rotate image by `degrees`."
    angle = degrees * math.pi / 180
    return [[cos(angle), -sin(angle), 0.],
            [sin(angle),  cos(angle), 0.],
            [0.        ,  0.        , 1.]]

def _get_zoom_mat(sw:float, sh:float, c:float, r:float)->AffineMatrix:
    "`sw`,`sh` scale width,height - `c`,`r` focus col,row."
    return [[sw, 0,  c],
            [0, sh,  r],
            [0,  0, 1.]]

@TfmAffine
def zoom(scale:uniform=1.0, row_pct:uniform=0.5, col_pct:uniform=0.5):
    "Zoom image by `scale`. `row_pct`,`col_pct` select focal point of zoom."
    s = 1-1/scale
    col_c = s * (2*col_pct - 1)
    row_c = s * (2*row_pct - 1)
    return _get_zoom_mat(1/scale, 1/scale, col_c, row_c)

@TfmAffine
def squish(scale:uniform=1.0, row_pct:uniform=0.5, col_pct:uniform=0.5):
    "Squish image by `scale`. `row_pct`,`col_pct` select focal point of zoom."
    if scale <= 1:
        col_c = (1-scale) * (2*col_pct - 1)
        return _get_zoom_mat(scale, 1, col_c, 0.)
    else:
        row_c = (1-1/scale) * (2*row_pct - 1)
        return _get_zoom_mat(1, 1/scale, 0., row_c)

@TfmCoord
def jitter(c, magnitude:uniform):
    "Replace pixels by random neighbors at `magnitude`."
    c.flow.add_((torch.rand_like(c.flow)-0.5)*magnitude*2)
    return c

@TfmPixel
def flip_lr(x): return x.flip(2)

@TfmAffine
def flip_affine() -> TfmAffine:
    return [[-1, 0, 0.],
            [0,  1, 0],
            [0,  0, 1.]]

@TfmPixel
def dihedral(x, k:partial(uniform_int,0,8)):
    "Randomly flip `x` image based on `k`."
    flips=[]
    if k&1: flips.append(1)
    if k&2: flips.append(2)
    if flips: x = torch.flip(x,flips)
    if k&4: x = x.transpose(1,2)
    return x.contiguous()

@TfmAffine
def dihedral_affine(k:partial(uniform_int,0,8)):
    "Randomly flip `x` image based on `k`."
    x = -1 if k&1 else 1
    y = -1 if k&2 else 1
    if k&4: return [[0, x, 0.],
                    [y, 0, 0],
                    [0, 0, 1.]]
    return [[x, 0, 0.],
            [0, y, 0],
            [0, 0, 1.]]

def _pad_coord(x, row_pad:int, col_pad:int, mode='zeros'):
    #TODO: implement other padding modes than zeros?
    h,w = x.size
    pad = torch.Tensor([w/(w + 2*col_pad), h/(h + 2*row_pad)])
    x.flow = FlowField((h+2*row_pad, w+2*col_pad) , x.flow.flow * pad[None])
    return x

@partial(TfmPixel, order=-10)
@singledispatch
def pad(x, padding:int, mode='reflection'):
    "Pad `x` with `padding` pixels. `mode` fills in space ('zeros','reflection','border')."
    mode = _pad_mode_convert[mode]
    return F.pad(x[None], (padding,)*4, mode=mode)[0]

@pad.register(ImagePoints)
def _(x, padding:int, mode='reflection'):
    return _pad_coord(x, padding, padding, mode)

@TfmPixel
@singledispatch
def crop(x, size, row_pct:uniform=0.5, col_pct:uniform=0.5):
    "Crop `x` to `size` pixels. `row_pct`,`col_pct` select focal point of crop."
    size = listify(size,2)
    rows,cols = size
    row = int((x.size(1)-rows+1) * row_pct)
    col = int((x.size(2)-cols+1) * col_pct)
    return x[:, row:row+rows, col:col+cols].contiguous()

@crop.register(ImagePoints)
def _(x, size, row_pct=0.5, col_pct=0.5):
    h,w = x.size
    rows,cols = listify(size, 2)
    x.flow.flow.mul_(torch.Tensor([w/cols, h/rows])[None])
    row = int((h-rows+1) * row_pct)
    col = int((w-cols+1) * col_pct)
    x.flow.flow.add_(-1 + torch.Tensor([w/cols-2*col/cols, h/rows-2*row/rows])[None])
    x.size = (rows, cols)
    return x

@TfmCrop
@singledispatch
def crop_pad(x, size, padding_mode='reflection',
             row_pct:uniform = 0.5, col_pct:uniform = 0.5):
    "Crop and pad tfm - `row_pct`,`col_pct` sets focal point."
    padding_mode = _pad_mode_convert[padding_mode]
    size = listify(size,2)
    if x.shape[1:] == size: return x
    rows,cols = size
    if x.size(1)<rows or x.size(2)<cols:
        row_pad = max((rows-x.size(1)+1)//2, 0)
        col_pad = max((cols-x.size(2)+1)//2, 0)
        x = F.pad(x[None], (col_pad,col_pad,row_pad,row_pad), mode=padding_mode)[0]
    row = int((x.size(1)-rows+1)*row_pct)
    col = int((x.size(2)-cols+1)*col_pct)

    x = x[:, row:row+rows, col:col+cols]
    return x.contiguous() # without this, get NaN later - don't know why

@crop_pad.register(ImagePoints)
def _(x, size, padding_mode='reflection', row_pct = 0.5, col_pct = 0.5):
    size = listify(size,2)
    rows,cols = size
    if x.size[0]<rows or x.size[1]<cols:
        row_pad = max((rows-x.size[0]+1)//2, 0)
        col_pad = max((cols-x.size[1]+1)//2, 0)
        x = _pad_coord(x, row_pad, col_pad)
    return crop(x,(rows,cols), row_pct, col_pct)

rand_pos = {'row_pct':(0,1), 'col_pct':(0,1)}

def rand_pad(padding:int, size:int, mode:str='reflection'):
    "Fixed `mode` `padding` and random crop of `size`"
    return [pad(padding=padding,mode=mode),
            crop(size=size, **rand_pos)]

def rand_zoom(*args, **kwargs):
    "Randomized version of `zoom`."
    return zoom(*args, **rand_pos, **kwargs)

def rand_crop(*args, **kwargs):
    "Randomized version of `crop_pad`."
    return crop_pad(*args, **rand_pos, **kwargs)

def zoom_crop(scale:float, do_rand:bool=False, p:float=1.0):
    "Randomly zoom and/or crop."
    zoom_fn = rand_zoom if do_rand else zoom
    crop_fn = rand_crop if do_rand else crop_pad
    return [zoom_fn(scale=scale, p=p), crop_fn()]

def _find_coeffs(orig_pts:Points, targ_pts:Points)->Tensor:
    "Find 8 coeff mentioned [here](https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/)."
    matrix = []
    #The equations we'll need to solve.
    for p1, p2 in zip(targ_pts, orig_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = FloatTensor(matrix)
    B = FloatTensor(orig_pts).view(8)
    #The 8 scalars we seek are solution of AX = B
    return torch.gesv(B,A)[0][:,0]

def _apply_perspective(coords:FlowField, coeffs:Points)->FlowField:
    "Transform `coords` with `coeffs`."
    size = coords.flow.size()
    #compress all the dims expect the last one ang adds ones, coords become N * 3
    coords.flow = coords.flow.view(-1,2)
    #Transform the coeffs in a 3*3 matrix with a 1 at the bottom left
    coeffs = torch.cat([coeffs, FloatTensor([1])]).view(3,3)
    coords.flow = torch.addmm(coeffs[:,2], coords.flow, coeffs[:,:2].t())
    coords.flow.mul_(1/coords.flow[:,2].unsqueeze(1))
    coords.flow = coords.flow[:,:2].view(size)
    return coords

_orig_pts = [[-1,-1], [-1,1], [1,-1], [1,1]]

def _perspective_warp(c:FlowField, targ_pts:Points, invert=False):
    "Apply warp to `targ_pts` from `_orig_pts` to `c` `FlowField`."
    if invert: return _apply_perspective(c, _find_coeffs(targ_pts, _orig_pts))
    return _apply_perspective(c, _find_coeffs(_orig_pts, targ_pts))

@TfmCoord
def perspective_warp(c, magnitude:partial(uniform,size=8)=0, invert=False):
    "Apply warp of `magnitude` to `c`."
    magnitude = magnitude.view(4,2)
    targ_pts = [[x+m for x,m in zip(xs, ms)] for xs, ms in zip(_orig_pts, magnitude)]
    return _perspective_warp(c, targ_pts, invert)

@TfmCoord
def symmetric_warp(c, magnitude:partial(uniform,size=4)=0, invert=False):
    "Apply symmetric warp of `magnitude` to `c`."
    m = listify(magnitude, 4)
    targ_pts = [[-1-m[3],-1-m[1]], [-1-m[2],1+m[1]], [1+m[3],-1-m[0]], [1+m[2],1+m[0]]]
    return _perspective_warp(c, targ_pts, invert)

@TfmCoord
def tilt(c, direction:uniform_int, magnitude:uniform=0, invert=False):
    "Tilt `c` field with random `direction` and `magnitude`."
    orig_pts = [[-1,-1], [-1,1], [1,-1], [1,1]]
    if direction == 0:   targ_pts = [[-1,-1], [-1,1], [1,-1-magnitude], [1,1+magnitude]]
    elif direction == 1: targ_pts = [[-1,-1-magnitude], [-1,1+magnitude], [1,-1], [1,1]]
    elif direction == 2: targ_pts = [[-1,-1], [-1-magnitude,1], [1,-1], [1+magnitude,1]]
    elif direction == 3: targ_pts = [[-1-magnitude,-1], [-1,1], [1+magnitude,-1], [1,1]]
    coeffs = _find_coeffs(targ_pts, _orig_pts) if invert else _find_coeffs(_orig_pts, targ_pts)
    return _apply_perspective(c, coeffs)

@TfmCoord
def skew(c, direction:uniform_int, magnitude:uniform=0, invert=False):
    "Skew `c` field with random `direction` and `magnitude`."
    orig_pts = [[-1,-1], [-1,1], [1,-1], [1,1]]
    if direction == 0:   targ_pts = [[-1-magnitude,-1], [-1,1], [1,-1], [1,1]]
    elif direction == 1: targ_pts = [[-1,-1-magnitude], [-1,1], [1,-1], [1,1]]
    elif direction == 2: targ_pts = [[-1,-1], [-1-magnitude,1], [1,-1], [1,1]]
    elif direction == 3: targ_pts = [[-1,-1], [-1,1+magnitude], [1,-1], [1,1]]
    elif direction == 4: targ_pts = [[-1,-1], [-1,1], [1+magnitude,-1], [1,1]]
    elif direction == 5: targ_pts = [[-1,-1], [-1,1], [1,-1-magnitude], [1,1]]
    elif direction == 6: targ_pts = [[-1,-1], [-1,1], [1,-1], [1+magnitude,1]]
    elif direction == 7: targ_pts = [[-1,-1], [-1,1], [1,-1], [1,1+magnitude]]
    coeffs = _find_coeffs(targ_pts, _orig_pts) if invert else _find_coeffs(_orig_pts, targ_pts)
    return _apply_perspective(c, coeffs)

def get_transforms(do_flip:bool=True, flip_vert:bool=False, max_rotate:float=10., max_zoom:float=1.1,
                   max_lighting:float=0.2, max_warp:float=0.2, p_affine:float=0.75,
                   p_lighting:float=0.75, xtra_tfms:float=None)->Collection[Transform]:
    "Utility func to easily create a list of flip, rotate, `zoom`, warp, lighting transforms."
    res = [rand_crop()]
    if do_flip:    res.append(dihedral_affine() if flip_vert else flip_affine(p=0.5))
    if max_warp:   res.append(symmetric_warp(magnitude=(-max_warp,max_warp), p=p_affine))
    if max_rotate: res.append(rotate(degrees=(-max_rotate,max_rotate), p=p_affine))
    if max_zoom>1: res.append(rand_zoom(scale=(1.,max_zoom), p=p_affine))
    if max_lighting:
        res.append(brightness(change=(0.5*(1-max_lighting), 0.5*(1+max_lighting)), p=p_lighting))
        res.append(contrast(scale=(1-max_lighting, 1/(1-max_lighting)), p=p_lighting))
    #       train                   , valid
    return (res + listify(xtra_tfms), [crop_pad()])

def _compute_zs_mat(sz:TensorImageSize, scale:float, squish:float,
                   invert:bool, row_pct:float, col_pct:float)->AffineMatrix:
    "Utility routine to compute zoom/squish matrix."
    orig_ratio = math.sqrt(sz[1]/sz[0])
    for s,r,i in zip(scale,squish, invert):
        s,r = 1/math.sqrt(s),math.sqrt(r)
        if s * r <= 1 and s / r <= 1: #Test if we are completely inside the picture
            w,h = (s/r, s*r) if i else (s*r,s/r)
            col_c = (1-w) * (2*col_pct - 1)
            row_c = (1-h) * (2*row_pct - 1)
            return _get_zoom_mat(w, h, col_c, row_c)

    #Fallback, hack to emulate a center crop without cropping anything yet.
    if orig_ratio > 1: return _get_zoom_mat(1/orig_ratio**2, 1, 0, 0.)
    else:              return _get_zoom_mat(1, orig_ratio**2, 0, 0.)

@TfmCoord
def zoom_squish(c, scale:uniform=1.0, squish:uniform=1.0, invert:rand_bool=False,
                row_pct:uniform=0.5, col_pct:uniform=0.5):
    #This is intended for scale, squish and invert to be of size 10 (or whatever) so that the transform
    #can try a few zoom/squishes before falling back to center crop (like torchvision.RandomResizedCrop)
    m = _compute_zs_mat(c.size, scale, squish, invert, row_pct, col_pct)
    return _affine_mult(c, FloatTensor(m))

def rand_resize_crop(size:int, max_scale:float=2., ratios:Tuple[float,float]=(0.75,1.33)):
    "Randomly resize and crop the image to a ratio in `ratios` after a zoom of `max_scale`."
    return [zoom_squish(scale=(1.,max_scale,8), squish=(*ratios,8), invert=(0.5,8), row_pct=(0.,1.), col_pct=(0.,1.)), 
            crop(size=size)]

