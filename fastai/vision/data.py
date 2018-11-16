"Manages data input pipeline - folderstransformbatch input. Includes support for classification, segmentation and bounding boxes"
from ..torch_core import *
from .image import *
from .transform import *
from ..data_block import *
from ..basic_data import *
from ..layers import *
from .learner import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import PIL

__all__ = ['get_image_files', 'denormalize', 'get_annotations', 'ImageDataBunch',
           'ImageItemList', 'normalize', 'normalize_funcs', 
           'channel_view', 'mnist_stats', 'cifar_stats', 'imagenet_stats', 'download_images',
           'verify_images', 'bb_pad_collate', 'ObjectCategoryProcessor',
           'ObjectCategoryList', 'ObjectItemList', 'SegmentationLabelList', 'SegmentationItemList', 'PointsItemList']

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(image_extensions if check_ext else None), recurse=recurse)

def get_annotations(fname, prefix=None):
    "Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
    annot_dict = json.load(open(fname))
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    classes = {}
    for o in annot_dict['categories']:
        classes[o['id']] = o['name']
    for o in annot_dict['annotations']:
        bb = o['bbox']
        id2bboxes[o['image_id']].append([bb[1],bb[0], bb[3]+bb[1], bb[2]+bb[0]])
        id2cats[o['image_id']].append(classes[o['category_id']])
    for o in annot_dict['images']:
        if o['id'] in id2bboxes:
            id2images[o['id']] = ifnone(prefix, '') + o['file_name']
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]

def bb_pad_collate(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data
        bboxes[i,-len(lbls):] = bbs
        labels[i,-len(lbls):] = lbls
    return torch.cat(imgs,0), (bboxes,labels)

def _maybe_add_crop_pad(tfms):
    tfm_names = [tfm.__name__ for tfm in tfms]
    return [crop_pad()] + tfms if 'crop_pad' not in tfm_names else tfms

def _prep_tfm_kwargs(tfms, kwargs):
    default_rsz = ResizeMethod.SQUISH if ('size' in kwargs and is_listy(kwargs['size'])) else ResizeMethod.CROP
    resize_method = ifnone(kwargs.get('resize_method', default_rsz), default_rsz)
    if resize_method <= 2: tfms = _maybe_add_crop_pad(tfms)
    kwargs['resize_method'] = resize_method
    return tfms, kwargs

def normalize(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Normalize `x` with `mean` and `std`."
    return (x-mean[...,None,None]) / std[...,None,None]

def denormalize(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return x*std[...,None,None] + mean[...,None,None]

def _normalize_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x.device),std.to(x.device)
    x = normalize(x,mean,std)
    if do_y: y = normalize(y,mean,std)
    return x,y

def normalize_funcs(mean:FloatTensor, std:FloatTensor)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_batch, mean=mean, std=std),
            partial(denormalize,      mean=mean, std=std))

cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mnist_stats = ([0.15]*3, [0.15]*3)

def channel_view(x:Tensor)->Tensor:
    "Make channel the first axis of `x` and flatten remaining axes"
    return x.transpose(0,1).contiguous().view(x.shape[1],-1)

def _get_fns(ds, path):
    "List of all file names relative to `path`."
    return [str(fn.relative_to(path)) for fn in ds.x]

class ImageDataBunch(DataBunch):
    @classmethod
    def create_from_ll(cls, dss:LabelLists, bs:int=64, ds_tfms:Optional[TfmList]=None,
                num_workers:int=defaults.cpus, tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
                test:Optional[PathOrStr]=None, collate_fn:Callable=data_collate, size:int=None, **kwargs)->'ImageDataBunch':
        dss = dss.transform(tfms=ds_tfms, size=size, **kwargs)
        if test is not None: dss.add_test_folder(test)
        return dss.databunch(bs=bs, tfms=tfms, num_workers=num_workers, collate_fn=collate_fn, device=device)

    @classmethod
    def from_folder(cls, path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid',
                    valid_pct=None, classes:Collection=None, **kwargs:Any)->'ImageDataBunch':
        "Create from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders (or provide `valid_pct`)."
        path=Path(path)
        il = ImageItemList.from_folder(path)
        if valid_pct is None: src = il.split_by_folder(train=train, valid=valid)
        else: src = il.random_split_by_pct(valid_pct)
        src = src.label_from_folder(classes=classes)
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, df:pd.DataFrame, folder:PathOrStr='.', sep=None, valid_pct:float=0.2,
                fn_col:IntsOrStrs=0, label_col:IntsOrStrs=1, suffix:str='',
                **kwargs:Any)->'ImageDataBunch':
        "Create from a DataFrame."
        src = (ImageItemList.from_df(df, path=path, folder=folder, suffix=suffix, cols=fn_col)
                .random_split_by_pct(valid_pct)
                .label_from_df(sep=sep, cols=label_col))
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, folder:PathOrStr='.', sep=None, csv_labels:PathOrStr='labels.csv', valid_pct:float=0.2,
            fn_col:int=0, label_col:int=1, suffix:str='',
            header:Optional[Union[int,str]]='infer', **kwargs:Any)->'ImageDataBunch':
        "Create from a csv file."
        path = Path(path)
        df = pd.read_csv(path/csv_labels, header=header)
        return cls.from_df(path, df, folder=folder, sep=sep, valid_pct=valid_pct,
                fn_col=fn_col, label_col=label_col, suffix=suffix, header=header, **kwargs)

    @classmethod
    def from_lists(cls, path:PathOrStr, fnames:FilePathList, labels:Collection[str], valid_pct:float=0.2, **kwargs):
        src = ImageItemList(fnames, path=path).random_split_by_pct(valid_pct).label_from_list(labels)
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_name_func(cls, path:PathOrStr, fnames:FilePathList, label_func:Callable, valid_pct:float=0.2, **kwargs):
        src = ImageItemList(fnames, path=path).random_split_by_pct(valid_pct)
        return cls.create_from_ll(src.label_from_func(label_func), **kwargs)

    @classmethod
    def from_name_re(cls, path:PathOrStr, fnames:FilePathList, pat:str, valid_pct:float=0.2, **kwargs):
        pat = re.compile(pat)
        def _get_label(fn): return pat.search(str(fn)).group(1)
        return cls.from_name_func(path, fnames, _get_label, valid_pct=valid_pct, **kwargs)

    def batch_stats(self, funcs:Collection[Callable]=None)->Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean,torch.std])
        x = self.valid_dl.one_batch()[0].cpu()
        return [func(channel_view(x), 1) for func in funcs]

    def normalize(self, stats:Collection[Tensor]=None)->None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        if stats is None: self.stats = self.batch_stats()
        else:             self.stats = stats
        self.norm,self.denorm = normalize_funcs(*self.stats)
        self.add_tfm(self.norm)
        return self

    def labels_to_csv(self, dest:str)->None:
        "Save file names and labels in `data` as CSV to file name `dest`."
        fns = _get_fns(self.train_ds)
        y = list(self.train_ds.y)
        fns += _get_fns(self.valid_ds)
        y += list(self.valid_ds.y)
        if hasattr(self,'test_dl') and data.test_dl:
            fns += _get_fns(self.test_ds)
            y += list(self.test_ds.y)
        df = pd.DataFrame({'name': fns, 'label': y})
        df.to_csv(dest, index=False)

    @staticmethod
    def single_from_classes(path:Union[Path, str], classes:Collection[str], tfms:TfmList=None, **kwargs):
        "Create an empty `ImageDataBunch` in `path` with `classes`. Typically used for inference."
        sd = ImageItemList([], path=path).split_by_idx([])
        return sd.label_const(0, label_cls=CategoryList, classes=classes).transform(tfms, **kwargs).databunch()

def download_image(url,dest, timeout=4):
    try: r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e: print(f"Error {url} {e}")

def download_images(urls:Collection[str], dest:PathOrStr, max_pics:int=1000, max_workers:int=8, timeout=4):
    "Download images listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_pics]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)

    if max_workers:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            suffixes = [re.findall(r'\.\w+?(?=(?:\?|$))', url) for url in urls]
            suffixes = [suffix[0] if len(suffix)>0  else '.jpg' for suffix in suffixes]
            futures = [ex.submit(download_image, url, dest/f"{i:08d}{suffixes[i]}", timeout=timeout)
                       for i,url in enumerate(urls)]
            for f in progress_bar(as_completed(futures), total=len(urls)): pass
    else:
        for i,url in enumerate(progress_bar(urls)):
            download_image(url, dest/f"{i:08d}.jpg", timeout=timeout)

def verify_image(file:Path, delete:bool, max_size:Union[int,Tuple[int,int]]=None, dest:Path=None, n_channels:int=3,
                 interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None, resume:bool=False, **kwargs):
    """Check if the image in `file` exists, can be opend and has `n_channels`. If `delete`, removes it if it fails.
    If `max_size` is specifided, image is resized to the same ratio so that both sizes are less than `max_size`,
    using `interp`. Result is stored in `dest`, `ext` forces an extension type, `img_format` and `kwargs` are passed
    to PIL.Image.save."""
    try:
        img = PIL.Image.open(file)
        if max_size is None: return
        assert isinstance(dest, Path), "You should provide `dest` Path to save resized image"
        max_size = listify(max_size, 2)
        if img.height > max_size[0] or img.width > max_size[1]:
            dest_fname = dest/file.name
            if ext is not None: dest_fname=dest_fname.with_suffix(ext)
            if resume and os.path.isfile(dest_fname): return
            ratio = img.height/img.width
            new_h = min(max_size[0], int(max_size[1] * ratio))
            new_w = int(new_h/ratio)
            if n_channels == 3: img = img.convert("RGB")
            img = img.resize((new_w,new_h), resample=interp)
            img.save(dest_fname, img_format, **kwargs)
        img = np.array(img)
        img_channels = 1 if len(img.shape) == 2 else img.shape[2]
        assert img_channels == n_channels, f"Image {file} has {img_channels} instead of {n_channels}"
    except Exception as e:
        print(f'{e}')
        if delete: file.unlink()

def verify_images(path:PathOrStr, delete:bool=True, max_workers:int=4, max_size:Union[int,Tuple[int,int]]=None,
                  dest:PathOrStr='.', n_channels:int=3, interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None,
                  resume:bool=None, **kwargs):
    """Check if the image in `path` exists, can be opened and has `n_channels`.
    If `n_channels` is 3 – it'll try to convert image to RGB. If `delete`, removes it if it fails.
    If `resume` – it will skip already existent images in `dest`.  If `max_size` is specifided,
    image is resized to the same ratio so that both sizes are less than `max_size`, using `interp`.
    Result is stored in `dest`, `ext` forces an extension type, `img_format` and `kwargs` are
    passed to PIL.Image.save. Use `max_workers` CPUs."""
    path = Path(path)
    if resume is None and dest == '.': resume=False
    dest = path/Path(dest)
    os.makedirs(dest, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        files = get_image_files(path)
        futures = [ex.submit(verify_image, file, delete=delete, max_size=max_size, dest=dest, n_channels=n_channels,
                             interp=interp, ext=ext, img_format=img_format, resume=resume, **kwargs) for file in files]
        for f in progress_bar(as_completed(futures), total=len(files)): pass

class ImageItemList(ItemList):
    _bunch = ImageDataBunch

    def __post_init__(self):
        super().__post_init__()
        self.sizes={}
        self.create_func = ifnone(self.create_func, open_image)

    def get(self, i):
        res = super().get(i)
        self.sizes[i] = res.size
        return res

    @classmethod
    def from_folder(cls, path:PathOrStr='.', create_func:Callable=open_image,
                    extensions:Collection[str]=image_extensions, **kwargs)->ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        return super().from_folder(create_func=create_func, path=path, extensions=extensions, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, create_func:Callable=open_image, cols:IntsOrStrs=0,
                 folder:PathOrStr='.', suffix:str='')->'ItemList':
        """Get the filenames in `col` of `df` and will had `path/folder` in front of them, `suffix` at the end.
        `create_func` is used to open the images."""
        suffix = suffix or ''
        res = super().from_df(df, path=path, create_func=create_func, cols=cols)
        res.items = np.char.add(np.char.add(f'{folder}/', res.items.astype(str)), suffix)
        res.items = np.char.add(f'{res.path}/', res.items)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, create_func:Callable=open_image, cols:IntsOrStrs=0, header:str='infer',
                 folder:PathOrStr='.', suffix:str='')->'ItemList':
        df = pd.read_csv(path/csv_name, header=header)
        return cls.from_df(df, path=path, create_func=create_func, cols=cols, folder=folder, suffix=suffix)

class ObjectCategoryProcessor(MultiCategoryProcessor):
    def process_one(self,item): return [item[0], [self.c2i.get(o,None) for o in item[1]]]

    def generate_classes(self, items):
        classes = super().generate_classes([o[1] for o in items])
        classes = ['background'] + list(classes)
        return classes

class ObjectCategoryList(MultiCategoryList):
    _processor = ObjectCategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)

    def get(self, i):
        return ImageBBox.create(*self.x.sizes[i], *self.items[i], classes=self.classes)

class ObjectItemList(ImageItemList):
    def __post_init__(self):
        super().__post_init__()
        self._label_cls = ObjectCategoryList

class SegmentationLabelList(ImageItemList):
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.classes,self.loss_func,self.create_func = classes,CrossEntropyFlat(),open_mask
        self.c = len(self.classes)

    def new(self, items, classes=None, **kwargs):
        return self.__class__(items, ifnone(classes, self.classes), **kwargs)

class SegmentationItemList(ImageItemList):
    def __post_init__(self):
        super().__post_init__()
        self._label_cls = SegmentationLabelList

class PointsItemList(ItemList):
    def __post_init__(self):
        super().__post_init__()
        self.c = len(self.items[0].view(-1))
        self.loss_func = MSELossFlat()

    def get(self, i):
        o = super().get(i)
        return ImagePoints(FlowField(self.x.sizes[i], o), scale=True)

