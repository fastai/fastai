"`Image` provides support to convert, transform and show images"
from ..torch_core import *
from ..basic_data import *
from io import BytesIO
import PIL

__all__ = ['Image', 'ImageBBox', 'ImageSegment', 'ImagePoints', 'FlowField', 'RandTransform', 'TfmAffine', 'TfmCoord',
           'TfmCrop', 'TfmLighting', 'TfmPixel', 'Transform', 'bb2hw', 'image2np', 'log_uniform',
           'logit', 'logit_', 'open_image', 'open_mask', 'pil2tensor', 'rand_bool', 'scale_flow', 'show_image',
           'uniform', 'uniform_int', 'CoordFunc', 'TfmList', 'open_mask_rle', 'rle_encode', 'rle_decode', 'ResizeMethod']

ResizeMethod = IntEnum('ResizeMethod', 'CROP PAD SQUISH NO')

def logit(x:Tensor)->Tensor:
    "Logit of `x`, clamped to avoid inf"
    x = x.clamp(1e-7, 1-1e-7)
    return -(1/x-1).log()

def logit_(x:Tensor)->Tensor:
    "Inplace logit of `x`, clamped to avoid inf"
    x.clamp_(1e-7, 1-1e-7)
    return (x.reciprocal_().sub_(1)).log_().neg_()

def uniform(low:Number, high:Number=None, size:Optional[List[int]]=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=`low`, max=`high`."
    if high is None: high=low
    return random.uniform(low,high) if size is None else torch.FloatTensor(*listify(size)).uniform_(low,high)

def log_uniform(low, high, size:Optional[List[int]]=None)->FloatOrTensor:
    "Draw 1 or shape=`size` random floats from uniform dist: min=log(`low`), max=log(`high`)."
    res = uniform(log(low), log(high), size)
    return exp(res) if size is None else res.exp_()

def rand_bool(p:float, size:Optional[List[int]]=None)->BoolOrTensor:
    "Draw 1 or shape=`size` random booleans (True occuring probability `p`)."
    return uniform(0,1,size)<p

def uniform_int(low:int, high:int, size:Optional[List[int]]=None)->IntOrTensor:
    "Generate int or tensor `size` of ints between `low` and `high` (included)."
    return random.randint(low,high) if size is None else torch.randint(low,high+1,size)

def pil2tensor(image:Union[NPImage,NPArray],dtype:np.dtype)->TensorImage:
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy( a.astype(dtype, copy=False) )

def image2np(image:Tensor)->np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,0).numpy()
    return res[...,0] if res.shape[2]==1 else res

def bb2hw(a:Collection[int])->np.ndarray:
    "Convert bounding box points from (width,height,center) to (height,width,top,left)."
    return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])

def _draw_outline(o:Patch, lw:int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def _draw_rect(ax:plt.Axes, b:Collection[int], color:str='white', text=None, text_size=14):
    "Draw bounding box on `ax`."
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(*b[:2], text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)

def _get_default_args(func:Callable):
    return {k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty}

@dataclass
class FlowField():
    "Wrap together some coords `flow` with a `size`."
    size:Tuple[int,int]
    flow:Tensor

CoordFunc = Callable[[FlowField, ArgStar, KWArgs], LogitTensorImage]

class Image(ItemBase):
    "Support applying transforms to image data in `px`."
    def __init__(self, px:Tensor):
        "Create from raw tensor image data `px`."
        self._px = px
        self._logit_px=None
        self._flow=None
        self._affine_mat=None
        self.sample_kwargs = {}

    def set_sample(self, **kwargs)->'ImageBase':
        "Set parameters that control how we `grid_sample` the image after transforms are applied."
        self.sample_kwargs = kwargs
        return self

    def clone(self):
        "Mimic the behavior of torch.clone for `Image` objects."
        return self.__class__(self.px.clone())

    @property
    def shape(self)->Tuple[int,int,int]: return self._px.shape
    @property
    def size(self)->Tuple[int,int]: return self.shape[-2:]
    @property
    def device(self)->torch.device: return self._px.device

    def __repr__(self): return f'{self.__class__.__name__} {tuple(self.shape)}'
    def _repr_png_(self): return self._repr_image_format('png')
    def _repr_jpeg_(self): return self._repr_image_format('jpeg')

    def _repr_image_format(self, format_str):
        with BytesIO() as str_buffer:
            plt.imsave(str_buffer, image2np(self.px), format=format_str)
            return str_buffer.getvalue()

    def apply_tfms(self, tfms:TfmList, do_resolve:bool=True, xtra:Optional[Dict[Callable,dict]]=None,
                   size:Optional[Union[int,TensorImageSize]]=None, mult:int=32,
                   resize_method:ResizeMethod=ResizeMethod.CROP, padding_mode:str='reflection', **kwargs:Any)->TensorImage:
        "Apply all `tfms` - `do_resolve`: bind random args - `size`, `mult` used to crop/pad."
        if not (tfms or xtra or size): return self
        xtra = ifnone(xtra, {})
        tfms = sorted(listify(tfms), key=lambda o: o.tfm.order)
        if do_resolve: _resolve_tfms(tfms)
        x = self.clone()
        x.set_sample(padding_mode=padding_mode, **kwargs)
        if size is not None:
            crop_target = _get_crop_target(size, mult=mult)
            if resize_method in (ResizeMethod.CROP,ResizeMethod.PAD):
                target = _get_resize_target(x, crop_target, do_crop=(resize_method==ResizeMethod.CROP))
                x.resize(target)
            elif resize_method==ResizeMethod.SQUISH: x.resize((x.shape[0],) + crop_target)
        else: size = x.size
        size_tfms = [o for o in tfms if isinstance(o.tfm,TfmCrop)]
        for tfm in tfms:
            if tfm.tfm in xtra: x = tfm(x, **xtra[tfm.tfm])
            elif tfm in size_tfms:
                if resize_method in (ResizeMethod.CROP,ResizeMethod.PAD):
                    x = tfm(x, size=size, padding_mode=padding_mode)
            else: x = tfm(x)
        return x

    def refresh(self)->None:
        "Apply any logit, flow, or affine transfers that have been sent to the `Image`."
        if self._logit_px is not None:
            self._px = self._logit_px.sigmoid_()
            self._logit_px = None
        if self._affine_mat is not None or self._flow is not None:
            self._px = _grid_sample(self._px, self.flow, **self.sample_kwargs)
            self.sample_kwargs = {}
            self._flow = None
        return self

    def save(self, fn:PathOrStr):
        "Save the image to `fn`."
        x = image2np(self.data*255).astype(np.uint8)
        PIL.Image.fromarray(x).save(fn)

    @property
    def px(self)->TensorImage:
        "Get the tensor pixel buffer."
        self.refresh()
        return self._px
    @px.setter
    def px(self,v:TensorImage)->None:
        "Set the pixel buffer to `v`."
        self._px=v

    @property
    def flow(self)->FlowField:
        "Access the flow-field grid after applying queued affine transforms."
        if self._flow is None:
            self._flow = _affine_grid(self.shape)
        if self._affine_mat is not None:
            self._flow = _affine_mult(self._flow,self._affine_mat)
            self._affine_mat = None
        return self._flow

    @flow.setter
    def flow(self,v:FlowField): self._flow=v

    def lighting(self, func:LightingFunc, *args:Any, **kwargs:Any):
        "Equivalent to `image = sigmoid(func(logit(image)))`."
        self.logit_px = func(self.logit_px, *args, **kwargs)
        return self

    def pixel(self, func:PixelFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.px = func(image.px)`."
        self.px = func(self.px, *args, **kwargs)
        return self

    def coord(self, func:CoordFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.flow = func(image.flow, image.size)`."
        self.flow = func(self.flow, *args, **kwargs)
        return self

    def affine(self, func:AffineFunc, *args, **kwargs)->'Image':
        "Equivalent to `image.affine_mat = image.affine_mat @ func()`."
        m = tensor(func(*args, **kwargs)).to(self.device)
        self.affine_mat = self.affine_mat @ m
        return self

    def resize(self, size:Union[int,TensorImageSize])->'Image':
        "Resize the image to `size`, size can be a single int."
        assert self._flow is None
        if isinstance(size, int): size=(self.shape[0], size, size)
        self.flow = _affine_grid(size)
        return self

    @property
    def affine_mat(self)->AffineMatrix:
        "Get the affine matrix that will be applied by `refresh`."
        if self._affine_mat is None:
            self._affine_mat = torch.eye(3).to(self.device)
        return self._affine_mat
    @affine_mat.setter
    def affine_mat(self,v)->None: self._affine_mat=v

    @property
    def logit_px(self)->LogitTensorImage:
        "Get logit(image.px)."
        if self._logit_px is None: self._logit_px = logit_(self.px)
        return self._logit_px
    @logit_px.setter
    def logit_px(self,v:LogitTensorImage)->None: self._logit_px=v

    @property
    def data(self)->TensorImage:
        "Return this images pixels as a tensor."
        return self.px

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
              cmap:str='viridis', y:Any=None, **kwargs):
        ax = show_image(self, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize)
        if y is not None: y.show(ax=ax, **kwargs)
        if title is not None: ax.set_title(title)

    def show_batch(self, idxs:Collection[int], rows:int, ds:Dataset, figsize:Tuple[int,int]=(9,10), **kwargs)->None:
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in zip(idxs[:rows*rows], (axs.flatten() if rows > 1 else [axs])):
            x,y = ds[i]
            x.show(ax=ax, y=y, **kwargs)
        plt.tight_layout()

    def show_results(self, xys, preds, figsize:Tuple[int,int]=None):
        rows = len(xys)
        figsize = ifnone(figsize, (8,3*rows))
        _,axs = plt.subplots(rows, 2, figsize=figsize)
        axs[0,0].set_title('Predictions')
        axs[0,1].set_title('Ground truth')
        for i,(x,y) in enumerate(xys):
            x.show(ax=axs[i,1], y=y)
            pred = y.reconstruct_output(preds[i], x)
            x.show(ax=axs[i,0], y=pred)
        plt.tight_layout()

class ImageSegment(Image):
    "Support applying transforms to segmentation masks data in `px`."
    def lighting(self, func:LightingFunc, *args:Any, **kwargs:Any)->'Image': return self

    def refresh(self):
        self.sample_kwargs['mode'] = 'nearest'
        return super().refresh()

    @property
    def data(self)->TensorImage:
        "Return this image pixels as a `LongTensor`."
        return self.px.long()

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        cmap:str='tab20', alpha:float=0.5, **kwargs):
        ax = show_image(self, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize,
                        interpolation='nearest', alpha=alpha, vmin=0)
        if title: ax.set_title(title)

    def reconstruct_output(self, out, x): return self.__class__(out.argmax(dim=0)[None])

class ImagePoints(Image):
    "Support applying transforms to a `flow` of points."
    def __init__(self, flow:FlowField, scale:bool=True, y_first:bool=True):
        "Create from raw tensor image data `px`."
        if scale: flow = scale_flow(flow)
        if y_first: flow.flow = flow.flow.flip(1)
        self._flow = flow
        self._affine_mat = None
        self.flow_func = []
        self.sample_kwargs = {}
        self.transformed = False

    def clone(self):
        "Mimic the behavior of torch.clone for `Image` objects."
        return self.__class__(FlowField(self.size, self.flow.flow.clone()), scale=False, y_first=False)

    def reconstruct_output(self, out, x): return self.__class__(FlowField(x.size, out[None]), scale=False)

    @property
    def shape(self)->Tuple[int,int,int]: return (1, *self._flow.size)
    @property
    def size(self)->Tuple[int,int]: return self._flow.size
    @size.setter
    def size(self, sz:int): self._flow.size=sz
    @property
    def device(self)->torch.device: return self._flow.flow.device

    def __repr__(self): return f'{self.__class__.__name__} {tuple(self.size)}'

    @property
    def flow(self)->FlowField:
        "Access the flow-field grid after applying queued affine and coord transforms."
        if self._affine_mat is not None:
            self._flow = _affine_inv_mult(self._flow, self._affine_mat)
            self._affine_mat = None
            self.transformed = True
        if len(self.flow_func) != 0:
            for f in self.flow_func[::-1]: self._flow = f(self._flow)
            self.transformed = True
            self.flow_func = []
        return self._flow

    @flow.setter
    def flow(self,v:FlowField):  self._flow=v

    def coord(self, func:CoordFunc, *args, **kwargs)->'ImagePoints':
        "Put `func` with `args` and `kwargs` in `self.flow_func` for later."
        if 'invert' in kwargs: kwargs['invert'] = True
        else: warn(f"{func.__name__} isn't implemented for {self.__class__}.")
        self.flow_func.append(partial(func, *args, **kwargs))
        return self

    def lighting(self, func:LightingFunc, *args:Any, **kwargs:Any)->'ImagePoints': return self

    def pixel(self, func:PixelFunc, *args, **kwargs)->'ImagePoints':
        "Equivalent to `self = func_flow(self)`."
        self = func(self, *args, **kwargs)
        self.transformed=True
        return self

    def refresh(self) -> 'ImagePoints':
        return self

    def resize(self, size:Union[int,TensorImageSize]) -> 'ImagePoints':
        "Resize the image to `size`, size can be a single int."
        if isinstance(size, int): size=(1, size, size)
        self._flow.size = size[1:]
        return self

    @property
    def data(self)->Tensor:
        "Return the points associated to this object."
        flow = self.flow #This updates flow before we test if some transforms happened
        if self.transformed:
            if 'remove_out' not in self.sample_kwargs or self.sample_kwargs['remove_out']:
                flow = _remove_points_out(flow)
            self.transformed=False
        return flow.flow.flip(1)

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True, **kwargs):
        if ax is None: _,ax = plt.subplots(figsize=figsize)
        pnt = scale_flow(FlowField(self.size, self.data), to_unit=False).flow.flip(1)
        ax.scatter(pnt[:, 0], pnt[:, 1], s=10, marker='.', c='r')
        if hide_axis: ax.axis('off')
        if title: ax.set_title(title)

class ImageBBox(ImagePoints):
    "Support applying transforms to a `flow` of bounding boxes."
    def __init__(self, flow:FlowField, scale:bool=True, y_first:bool=True, labels:Collection=None,
                 classes:dict=None, pad_idx:int=0):
        super().__init__(flow, scale, y_first)
        self.pad_idx = pad_idx
        if labels is not None and len(labels)>0 and not isinstance(labels[0],Category):
            labels = array([Category(l,classes[l]) for l in labels])
        self.labels = labels

    def clone(self) -> 'ImageBBox':
        "Mimic the behavior of torch.clone for `Image` objects."
        flow = FlowField(self.size, self.flow.flow.clone())
        return self.__class__(flow, scale=False, y_first=False, labels=self.labels, pad_idx=self.pad_idx)

    @classmethod
    def create(cls, h:int, w:int, bboxes:Collection[Collection[int]], labels:Collection=None, classes:dict=None,
               pad_idx:int=0)->'ImageBBox':
        "Create an ImageBBox object from `bboxes`."
        bboxes = tensor(bboxes).float()
        tr_corners = torch.cat([bboxes[:,0][:,None], bboxes[:,3][:,None]], 1)
        bl_corners = bboxes[:,1:3].flip(1)
        bboxes = torch.cat([bboxes[:,:2], tr_corners, bl_corners, bboxes[:,2:]], 1)
        flow = FlowField((h,w), bboxes.view(-1,2))
        return cls(flow, labels=labels, classes=classes, pad_idx=pad_idx, y_first=True)

    def _compute_boxes(self) -> Tuple[LongTensor, LongTensor]:
        bboxes = self.flow.flow.flip(1).view(-1, 4, 2).contiguous().clamp(min=-1, max=1)
        mins, maxes = bboxes.min(dim=1)[0], bboxes.max(dim=1)[0]
        bboxes = torch.cat([mins, maxes], 1)
        mask = (bboxes[:,2]-bboxes[:,0] > 0) * (bboxes[:,3]-bboxes[:,1] > 0)
        if len(mask) == 0: return tensor([self.pad_idx] * 4), tensor([self.pad_idx])
        res = bboxes[mask]
        if self.labels is None: return res,None
        return res, self.labels[to_np(mask).astype(bool)]

    @property
    def data(self)->Union[FloatTensor, Tuple[FloatTensor,LongTensor]]:
        bboxes,lbls = self._compute_boxes()
        lbls = tensor([o.data for o in lbls]) if lbls is not None else None
        return bboxes if lbls is None else (bboxes, lbls)

    def show(self, y:Image=None, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        color:str='white', **kwargs):
        if ax is None: _,ax = plt.subplot(figsize=figsize)
        bboxes, lbls = self._compute_boxes()
        h,w = self.flow.size
        bboxes.add_(1).mul_(torch.tensor([h/2, w/2, h/2, w/2])).long()
        for i, bbox in enumerate(bboxes):
            if lbls is not None: text = str(lbls[i])
            else: text=None
            _draw_rect(ax, bb2hw(bbox), text=text, color=color)

def open_image(fn:PathOrStr)->Image:
    "Return `Image` object created from image in file `fn`."
    x = PIL.Image.open(fn).convert('RGB')
    return Image(pil2tensor(x,np.float32).div_(255))

def open_mask(fn:PathOrStr, div=False, convert_mode='L')->ImageSegment:
    "Return `ImageSegment` object create from mask in file `fn`. If `div`, divides pixel values by 255."
    x = PIL.Image.open(fn).convert(convert_mode)
    mask = pil2tensor(x,np.float32)
    if div: mask.div_(255)
    return ImageSegment(mask)

def open_mask_rle(mask_rle:str, shape:Tuple[int, int])->ImageSegment:
    "Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`."
    x = FloatTensor(rle_decode(str(mask_rle), shape).astype(np.uint8))
    x = x.view(shape[1], shape[0], -1)
    return ImageSegment(x.permute(2,0,1))

def rle_encode(img:NPArrayMask)->str:
    "Return run-length encoding string from an image array"
    pixels = np.concatenate([[0], img.flatten() , [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle:str, shape:Tuple[int,int])->NPArrayMask:
    "Return an image array from run-length encoded string"
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint)
    for low, up in zip(starts, ends): img[low:up] = 1
    return img.reshape(shape)

def show_image(img:Image, ax:plt.Axes=None, figsize:tuple=(3,3), hide_axis:bool=True, cmap:str='binary',
                alpha:float=None, **kwargs)->plt.Axes:
    "Display `Image` in notebook"
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(image2np(img.data), cmap=cmap, alpha=alpha, **kwargs)
    if hide_axis: ax.axis('off')
    return ax

def scale_flow(flow, to_unit=True):
    "Scale the coords in `flow` to -1/1 or the image size depending on `to_unit`."
    s = tensor([flow.size[0]/2,flow.size[1]/2])[None]
    if to_unit: flow.flow = flow.flow/s-1
    else:       flow.flow = (flow.flow+1)*s
    return flow

def _remove_points_out(flow:FlowField):
    pad_mask = (flow.flow[:,0] >= -1) * (flow.flow[:,0] <= 1) * (flow.flow[:,1] >= -1) * (flow.flow[:,1] <= 1)
    flow.flow = flow.flow[pad_mask]
    return flow

class Transform():
    "Utility class for adding probability and wrapping support to transform `func`."
    _wrap=None
    order=0
    def __init__(self, func:Callable, order:Optional[int]=None):
        "Create a transform for `func` and assign it an priority `order`, attach to `Image` class."
        if order is not None: self.order=order
        self.func=func
        functools.update_wrapper(self, self.func)
        self.func.__annotations__['return'] = Image
        self.params = copy(func.__annotations__)
        self.def_args = _get_default_args(func)
        setattr(Image, func.__name__,
                lambda x, *args, **kwargs: self.calc(x, *args, **kwargs))

    def __call__(self, *args:Any, p:float=1., is_random:bool=True, **kwargs:Any)->Image:
        "Calc now if `args` passed; else create a transform called prob `p` if `random`."
        if args: return self.calc(*args, **kwargs)
        else: return RandTransform(self, kwargs=kwargs, is_random=is_random, p=p)

    def calc(self, x:Image, *args:Any, **kwargs:Any)->Image:
        "Apply to image `x`, wrapping it if necessary."
        if self._wrap: return getattr(x, self._wrap)(self.func, *args, **kwargs)
        else:          return self.func(x, *args, **kwargs)

    @property
    def name(self)->str: return self.__class__.__name__

    def __repr__(self)->str: return f'{self.name} ({self.func.__name__})'

@dataclass
class RandTransform():
    "Wrap `Transform` to add randomized execution."
    tfm:Transform
    kwargs:dict
    p:int=1.0
    resolved:dict = field(default_factory=dict)
    do_run:bool = True
    is_random:bool = True
    def __post_init__(self): functools.update_wrapper(self, self.tfm)

    def resolve(self)->None:
        "Binds any random variables in the transform."
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # for each param passed to tfm...
        for k,v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else: self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k,v in self.tfm.def_args.items():
            if k not in self.resolved: self.resolved[k]=v
        # anything left over must be callable without params
        for k,v in self.tfm.params.items():
            if k not in self.resolved and k!='return': self.resolved[k]=v()

        self.do_run = rand_bool(self.p)

    @property
    def order(self)->int: return self.tfm.order

    def __call__(self, x:Image, *args, **kwargs)->Image:
        "Randomly execute our tfm on `x`."
        return self.tfm(x, *args, **{**self.resolved, **kwargs}) if self.do_run else x

def _resolve_tfms(tfms:TfmList):
    "Resolve every tfm in `tfms`."
    for f in listify(tfms): f.resolve()

def _grid_sample(x:TensorImage, coords:FlowField, mode:str='bilinear', padding_mode:str='reflection', **kwargs)->TensorImage:
    "Grab pixels in `coords` from `input` sampling by `mode`. `paddding_mode` is reflection, border or zeros."
    coords = coords.flow.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) # optimize layout for grid_sample
    return F.grid_sample(x[None], coords, mode=mode, padding_mode=padding_mode)[0]

def _affine_grid(size:TensorImageSize)->FlowField:
    size = ((1,)+size)
    N, C, H, W = size
    grid = FloatTensor(N, H, W, 2)
    linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1])
    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1])
    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
    return FlowField(size[2:], grid)

def _affine_mult(c:FlowField,m:AffineMatrix)->FlowField:
    "Multiply `c` by `m` - can adjust for rectangular shaped `c`."
    if m is None: return c
    size = c.flow.size()
    h,w = c.size
    m[0,1] *= h/w
    m[1,0] *= w/h
    c.flow = c.flow.view(-1,2)
    c.flow = torch.addmm(m[:2,2], c.flow,  m[:2,:2].t()).view(size)
    return c

def _affine_inv_mult(c, m):
    "Applies the inverse affine transform described in m"
    size = c.flow.size()
    h,w = c.size
    m[0,1] *= h/w
    m[1,0] *= w/h
    c.flow = c.flow.view(-1,2)
    a = torch.inverse(m[:2,:2].t())
    c.flow = torch.mm(c.flow - m[:2,2], a).view(size)
    return c

class TfmAffine(Transform):
    "Decorator for affine tfm funcs."
    order,_wrap = 5,'affine'
class TfmPixel(Transform):
    "Decorator for pixel tfm funcs."
    order,_wrap = 10,'pixel'
class TfmCoord(Transform):
    "Decorator for coord tfm funcs."
    order,_wrap = 4,'coord'
class TfmCrop(TfmPixel):
    "Decorator for crop tfm funcs."
    order=99
class TfmLighting(Transform):
    "Decorator for lighting tfm funcs."
    order,_wrap = 8,'lighting'

def _round_multiple(x:int, mult:int)->int:
    "Calc `x` to nearest multiple of `mult`."
    return (int(x/mult+0.5)*mult)

def _get_crop_target(target_px:Union[int,Tuple[int,int]], mult:int=32)->Tuple[int,int]:
    "Calc crop shape of `target_px` to nearest multiple of `mult`."
    target_r,target_c = listify(target_px, 2)
    return _round_multiple(target_r,mult),_round_multiple(target_c,mult)

def _get_resize_target(img, crop_target, do_crop=False)->TensorImageSize:
    "Calc size of `img` to fit in `crop_target` - adjust based on `do_crop`."
    if crop_target is None: return None
    ch,r,c = img.shape
    target_r,target_c = crop_target
    ratio = (min if do_crop else max)(r/target_r, c/target_c)
    return ch,round(r/ratio),round(c/ratio)

