"`vision.data` manages data input pipeline - folderstransformbatch input. Includes support for classification, segmentation and bounding boxes"
from ..torch_core import *
from .image import *
from .transform import *
from ..data_block import *
from ..data_block import _df_to_fns_labels
from ..basic_data import *
from ..layers import CrossEntropyFlat
from concurrent.futures import ProcessPoolExecutor, as_completed
import PIL

__all__ = ['get_image_files', 'DatasetTfm', 'ImageClassificationDataset', 'ImageMultiDataset', 'ObjectDetectDataset',
           'SegmentationDataset', 'ImageClassificationBase', 'denormalize', 'get_annotations', 'ImageDataBunch', 'ImageFileList', 'normalize',
           'normalize_funcs', 'show_image_batch', 'transform_datasets', 'SplitDatasetsImage', 'channel_view',
           'mnist_stats', 'cifar_stats', 'imagenet_stats', 'download_images', 'verify_images', 'bb_pad_collate']

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=image_extensions, recurse=recurse)

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

def show_image_batch(dl:DataLoader, classes:Collection[str], rows:int=None, figsize:Tuple[int,int]=(9,10))->None:
    "Show a few images from a batch."
    b_idx = next(iter(dl.batch_sampler))
    if rows is None: rows = int(math.sqrt(len(b_idx)))
    fig, axs = plt.subplots(rows,rows,figsize=figsize)
    for i, ax in zip(b_idx[:rows*rows], axs.flatten()):
        x,y = dl.dataset[i]
        x.show(ax=ax, y=y, classes=classes)
    plt.tight_layout()

class SplitDatasetsImage(SplitDatasets):
    def transform(self, tfms:TfmList, **kwargs)->'SplitDatasets':
        "Apply `tfms` to the underlying datasets, `kwargs` are passed to `DatasetTfm`."
        assert not isinstance(self.train_ds, DatasetTfm)
        self.train_ds = DatasetTfm(self.train_ds, tfms[0],  **kwargs)
        self.valid_ds = DatasetTfm(self.valid_ds, tfms[1],  **kwargs)
        if self.test_ds is not None:
            self.test_ds = DatasetTfm(self.test_ds, tfms[1],  **kwargs)
        return self

    def databunch(self, path:PathOrStr=None, **kwargs)->'ImageDataBunch':
        "Create an `ImageDataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `ImageDataBunch.create`."
        path = Path(ifnone(path, self.path))
        return ImageDataBunch.create(*self.datasets, path=path, **kwargs)

class ImageClassificationBase(LabelDataset):
    __splits_class__ = SplitDatasetsImage

    def __init__(self, fns:FilePathList, classes:Optional[Collection[Any]]=None):
        super().__init__(classes=classes)
        self.x  = np.array(fns)
        self.image_opener = open_image

    def _get_x(self,i): return self.image_opener(self.x[i])

    def new(self, *args, classes:Optional[Collection[Any]]=None, **kwargs):
        if classes is None: classes = self.classes
        res = self.__class__(*args, classes=classes, **kwargs)
        return res

class ImageClassificationDataset(ImageClassificationBase):
    "`Dataset` for folders of images in style {folder}/{class}/{images}."
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Collection[Any]]=None):
        if classes is None: classes = uniqueify(labels)
        super().__init__(fns, classes)
        self.y = np.array([self.class2idx[o] for o in labels], dtype=np.int64)
        self.loss_func = F.cross_entropy

    @staticmethod
    def _folder_files(folder:Path, label:ImgLabel, extensions:Collection[str]=image_extensions)->Tuple[FilePathList,ImgLabels]:
        "From `folder` return image files and labels. The labels are all `label`. Only keep files with suffix in `extensions`."
        fnames = get_files(folder, extensions=extensions)
        return fnames,[label]*len(fnames)

    @classmethod
    def from_single_folder(cls, folder:PathOrStr, classes:Collection[Any], extensions:Collection[str]=image_extensions):
        "Typically used for test set. Label all images in `folder`  with suffix in `extensions` with `classes[0]`."
        fns,labels = cls._folder_files(folder, classes[0], extensions=extensions)
        return cls(fns, labels, classes=classes)

    @classmethod
    def from_folder(cls, folder:Path, classes:Optional[Collection[Any]]=None, valid_pct:float=0.,
            extensions:Collection[str]=image_extensions)->Union['ImageClassificationDataset', List['ImageClassificationDataset']]:
        "Dataset of `classes` labeled images in `folder`. Optional `valid_pct` split validation set."
        if classes is None: classes = [cls.name for cls in find_classes(folder)]

        fns,labels,keep = [],[],{}
        for cl in classes:
            f,l = cls._folder_files(folder/cl, cl, extensions=extensions)
            fns+=f; labels+=l
            keep[cl] = len(f)
        classes = [cl for cl in classes if keep[cl]]

        if valid_pct==0.: return cls(fns, labels, classes=classes)
        return [cls(*a, classes=classes) for a in random_split(valid_pct, fns, labels)]

class ImageMultiDataset(ImageClassificationBase):
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Collection[Any]]=None):
        if classes is None: classes = uniqueify(np.concatenate(labels))
        super().__init__(fns, classes)
        self.y = [np.array([self.class2idx[o] for o in l], dtype=np.int64) for l in labels]
        self.loss_func = F.binary_cross_entropy_with_logits

    def encode(self, x:Collection[int]):
        "One-hot encode the target."
        res = np.zeros((self.c,), np.float32)
        res[x] = 1.
        return res

    def get_labels(self, idx:int)->ImgLabels: return [self.classes[i] for i in self.y[idx]]
    def _get_y(self,i): return self.encode(self.y[i])

    @classmethod
    def from_single_folder(cls, folder:PathOrStr, classes:Collection[Any], extensions=image_extensions):
        "Typically used for test set; label all images in `folder` with `classes[0]`."
        fnames = get_files(folder, extensions=extensions)
        labels = [[classes[0]]] * len(fnames)
        return cls(fnames, labels, classes=classes)

    @classmethod
    def from_folder(cls, path:PathOrStr, folder:PathOrStr, fns:pd.Series, labels:ImgLabels, valid_pct:float=0.2,
        classes:Optional[Collection[Any]]=None):
        path = Path(path)
        folder_path = (path/folder).absolute()
        train,valid = random_split(valid_pct, f'{folder_path}/' + fns, labels)
        train_ds = cls(*train, classes=classes)
        return [train_ds,cls(*valid, classes=train_ds.classes)]

class SegmentationDataset(ImageClassificationBase):
    "A dataset for segmentation task."
    def __init__(self, x:FilePathList, y:FilePathList, classes:Collection[Any]):
        assert len(x)==len(y)
        super().__init__(x, classes)
        self.y = np.array(y)
        self.loss_func = CrossEntropyFlat()
        self.mask_opener = open_mask

    def _get_y(self,i): return self.mask_opener(self.y[i])

class ObjectDetectDataset(ImageClassificationBase):
    "A dataset with annotated images."
    def __init__(self, x_fns:Collection[Path], labelled_bbs:Collection[Tuple[Collection[int], str]], classes:Collection[str]=None):
        assert len(x_fns)==len(labelled_bbs)
        if classes is None:
            classes = set()
            for lbl_bb in labelled_bbs: classes = classes.union(set(lbl_bb[1]))
            classes = ['background'] + list(classes)
        super().__init__(x_fns,classes)
        self.labelled_bbs = labelled_bbs

    def _get_y(self,i):
        #TODO: find a smart way to not reopen the x image.
        cats = LongTensor([self.class2idx[l] for l in self.labelled_bbs[i][1]])
        return (ImageBBox.create(self.labelled_bbs[i][0], *self._get_x(i).size, cats))

    @classmethod
    def from_json(cls, folder, fname, valid_pct=None, classes=None):
        """Create an `ObjectDetectDataset` by looking at the images in `folder` according to annotations in the json `fname`.
        If `valid_pct` is passed, split a training and validation set. `classes` is the list of classes."""
        imgs, labelled_bbox = get_annotations(fname, prefix=f'{folder}/')
        if valid_pct:
            train,valid = random_split(valid_pct, imgs, labelled_bbox)
            train_ds = cls(*train, classes=classes)
            return train_ds, cls(*valid, classes=train_ds.classes)
        return cls(imgs, labelled_bbox, classes=classes)

def bb_pad_collate(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect samples and adds padding."
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

class DatasetTfm(Dataset):
    "`Dataset` that applies a list of transforms to every item drawn."
    def __init__(self, ds:Dataset, tfms:TfmList=None, tfm_y:bool=False, **kwargs:Any):
        "this dataset will apply `tfms` to `ds`"
        self.ds,self.tfm_y = ds,tfm_y
        self.tfms,self.kwargs = _prep_tfm_kwargs(tfms,kwargs)
        self.y_kwargs = {**self.kwargs, 'do_resolve':False}

    def __len__(self)->int: return len(self.ds)
    def __repr__(self)->str: return f'{self.__class__.__name__}({self.ds})'

    def __getitem__(self,idx:int)->Tuple[ItemBase,Any]:
        "Return tfms(x),y."
        x,y = self.ds[idx]
        x = apply_tfms(self.tfms, x, **self.kwargs)
        if self.tfm_y: y = apply_tfms(self.tfms, y, **self.y_kwargs)
        return x, y

    def __getattr__(self,k):
        "Passthrough access to wrapped dataset attributes."
        return getattr(self.ds, k)

def _transform_dataset(self, tfms:TfmList=None, tfm_y:bool=False, **kwargs:Any)->DatasetTfm:
    return DatasetTfm(self, tfms=tfms, tfm_y=tfm_y, **kwargs)
DatasetBase.transform = _transform_dataset

def transform_datasets(train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None,
                       tfms:Optional[Tuple[TfmList,TfmList]]=None, resize_method:ResizeMethod=None, **kwargs:Any):
    "Create train, valid and maybe test DatasetTfm` using `tfms` = (train_tfms,valid_tfms)."
    tfms = ifnone(tfms, [[],[]])
    res = [DatasetTfm(train_ds, tfms[0], resize_method=resize_method, **kwargs),
           DatasetTfm(valid_ds, tfms[1], resize_method=resize_method, **kwargs)]
    if test_ds is not None: res.append(DatasetTfm(test_ds, tfms[1], resize_method=resize_method, **kwargs))
    return res

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
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, ds_tfms:Optional[TfmList]=None,
                     num_workers:int=defaults.cpus, tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
                     collate_fn:Callable=data_collate, size:int=None, **kwargs)->'ImageDataBunch':
        "Factory method. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        if ds_tfms or size: datasets = transform_datasets(*datasets, tfms=ds_tfms, size=size, **kwargs)
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,bs*2,bs*2), (True,False,False))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn)

    @classmethod
    def from_folder(cls, path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid',
                    test:Optional[PathOrStr]=None, valid_pct=None, **kwargs:Any)->'ImageDataBunch':
        "Create from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders (or provide `valid_pct`)."
        path=Path(path)
        if valid_pct is None:
            train_ds = ImageClassificationDataset.from_folder(path/train)
            datasets = [train_ds, ImageClassificationDataset.from_folder(path/valid, classes=train_ds.classes)]
        else: datasets = ImageClassificationDataset.from_folder(path/train, valid_pct=valid_pct)

        if test: datasets.append(ImageClassificationDataset.from_single_folder(
            path/test,classes=datasets[0].classes))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, df:pd.DataFrame, folder:PathOrStr='.', sep=None, valid_pct:float=0.2,
            fn_col:int=0, label_col:int=1, test:Optional[PathOrStr]=None, suffix:str=None, **kwargs:Any)->'ImageDataBunch':
        "Create from a DataFrame."
        path = Path(path)
        fnames, labels = _df_to_fns_labels(df, suffix=suffix, label_delim=sep, fn_col=fn_col, label_col=label_col)
        if sep:
            classes = uniqueify(np.concatenate(labels))
            datasets = ImageMultiDataset.from_folder(path, folder, fnames, labels, valid_pct=valid_pct, classes=classes)
            if test: datasets.append(ImageMultiDataset.from_single_folder(path/test, classes=datasets[0].classes))
        else:
            folder_path = (path/folder).absolute()
            (train_fns,train_lbls), (valid_fns,valid_lbls) = random_split(valid_pct, f'{folder_path}/' + fnames, labels)
            classes = uniqueify(labels)
            datasets = [ImageClassificationDataset(train_fns, train_lbls, classes)]
            datasets.append(ImageClassificationDataset(valid_fns, valid_lbls, classes))
            if test: datasets.append(ImageClassificationDataset.from_single_folder(Path(path)/test, classes=classes))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, folder:PathOrStr='.', sep=None, csv_labels:PathOrStr='labels.csv', valid_pct:float=0.2,
            fn_col:int=0, label_col:int=1, test:Optional[PathOrStr]=None, suffix:str=None,
            header:Optional[Union[int,str]]='infer', **kwargs:Any)->'ImageDataBunch':
        "Create from a csv file."
        path = Path(path)
        df = pd.read_csv(path/csv_labels, header=header)
        return cls.from_df(path, df, folder=folder, sep=sep, valid_pct=valid_pct, test=test,
                fn_col=fn_col, label_col=label_col, suffix=suffix, header=header, **kwargs)

    @classmethod
    def from_lists(cls, path:PathOrStr, fnames:FilePathList, labels:Collection[str], valid_pct:float=0.2, test:str=None, **kwargs):
        classes = uniqueify(labels)
        train,valid = random_split(valid_pct, fnames, labels)
        datasets = [ImageClassificationDataset(*train, classes),
                    ImageClassificationDataset(*valid, classes)]
        if test: datasets.append(ImageClassificationDataset.from_single_folder(Path(path)/test, classes=classes))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def from_name_func(cls, path:PathOrStr, fnames:FilePathList, label_func:Callable, valid_pct:float=0.2, test:str=None, **kwargs):
        labels = [label_func(o) for o in fnames]
        return cls.from_lists(path, fnames, labels, valid_pct=valid_pct, test=test, **kwargs)

    @classmethod
    def from_name_re(cls, path:PathOrStr, fnames:FilePathList, pat:str, valid_pct:float=0.2, test:str=None, **kwargs):
        pat = re.compile(pat)
        def _get_label(fn): return pat.search(str(fn)).group(1)
        return cls.from_name_func(path, fnames, _get_label, valid_pct=valid_pct, test=test, **kwargs)

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

    def show_batch(self:DataBunch, rows:int=None, figsize:Tuple[int,int]=(9,10), ds_type:DatasetType=DatasetType.Train)->None:
        show_image_batch(self.dl(ds_type), self.classes, figsize=figsize, rows=rows)

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
    def single_from_classes(path:Union[Path, str], classes:Collection[str], **kwargs):
        return SplitDatasetsImage.single_from_classes(path, classes).transform(**kwargs).databunch(bs=1)

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
            futures = [ex.submit(download_image, url, dest/f"{i:08d}.jpg", timeout=timeout)
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

class ImageFileList(InputList):
    "A list of inputs. Contain methods to get the corresponding labels."
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=image_extensions, recurse=True)->'ImageFileList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions=extensions, recurse=recurse), path)

def split_data_add_test_folder(self, test_folder:str='test', label:Any=None):
    "Add test set containing items from folder `test_folder` and an arbitrary label"
    items = ImageFileList.from_folder(self.path/test_folder)
    return self.add_test(items, label=label)

SplitData.add_test_folder = split_data_add_test_folder
