"`vision.data` manages data input pipeline - folderstransformbatch input. Includes support for classification, segmentation and bounding boxes"
from ..torch_core import *
from .image import *
from .transform import *
from ..data_block import *
from ..data_block import _df_to_fns_labels
from ..basic_data import *
from ..layers import CrossEntropyFlat
from concurrent.futures import ProcessPoolExecutor, as_completed

__all__ = ['get_image_files', 'DatasetTfm', 'ImageClassificationDataset', 'ImageMultiDataset', 'ObjectDetectDataset',
           'SegmentationDataset', 'ImageDataset', 'denormalize', 'get_annotations', 'ImageDataBunch', 'ImageFileList', 'normalize',
           'normalize_funcs', 'show_image_batch', 'transform_datasets', 'SplitDatasetsImage', 'channel_view',
           'mnist_stats', 'cifar_stats', 'imagenet_stats', 'download_images', 'verify_images', 'bb_pad_collate']

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=image_extensions)

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

class ImageDataset(LabelDataset):
    __splits_class__ = SplitDatasetsImage
    def _get_x(self,i): return open_image(self.x[i])

class ImageClassificationDataset(ImageDataset):
    "`Dataset` for folders of images in style {folder}/{class}/{images}."
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Collection[Any]]=None):
        self.x  = np.array(fns)
        if classes is None: classes = uniqueify(labels)
        super().__init__(classes)
        self.y = np.array([self.class2idx[o] for o in labels], dtype=np.int64)
        self.loss_func = F.cross_entropy

    def _get_y(self,i): return self.y[i]

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

class ImageMultiDataset(ImageDataset):
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Collection[Any]]=None):
        if classes is None: classes = uniqueify(np.concatenate(labels))
        super().__init__(classes)
        self.x = np.array(fns)
        self.y = [np.array([self.class2idx[o] for o in l], dtype=np.int64) for l in labels]
        self.loss_func = F.binary_cross_entropy_with_logits

    def encode(self, x:Collection[int]):
        "One-hot encode the target."
        res = np.zeros((self.c,), np.float32)
        res[x] = 1.
        return res

    def get_labels(self, idx:int)->ImgLabels: return [self.classes[i] for i in self.y[idx]]
    def _get_x(self,i): return open_image(self.x[i])
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

class SegmentationDataset(ImageDataset):
    "A dataset for segmentation task."
    def __init__(self, x:FilePathList, y:FilePathList, classes:Collection[Any], div=False, convert_mode='L'):
        assert len(x)==len(y)
        super().__init__(classes)
        self.x,self.y,self.div,self.convert_mode = np.array(x),np.array(y),div,convert_mode
        self.loss_func = CrossEntropyFlat()

    def _get_x(self,i): return open_image(self.x[i])
    def _get_y(self,i): return open_mask(self.y[i], self.div, self.convert_mode)

class ObjectDetectDataset(ImageDataset):
    "A dataset with annotated images."
    def __init__(self, x_fns:Collection[Path], labelled_bbs:Collection[Tuple[Collection[int], str]], classes:Collection[str]=None):
        assert len(x_fns)==len(labelled_bbs)
        if classes is None:
            classes = set()
            for lbl_bb in labelled_bbs: classes = classes.union(set(lbl_bb[1]))
            classes = ['background'] + list(classes)
        super().__init__(classes)
        self.x,self.labelled_bbs = x_fns,labelled_bbs

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

class DatasetTfm(Dataset):
    "`Dataset` that applies a list of transforms to every item drawn."
    def __init__(self, ds:Dataset, tfms:TfmList=None, tfm_y:bool=False, **kwargs:Any):
        "this dataset will apply `tfms` to `ds`"
        self.ds,self.tfms,self.kwargs,self.tfm_y = ds,tfms,kwargs,tfm_y
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
                       tfms:Optional[Tuple[TfmList,TfmList]]=None, **kwargs:Any):
    "Create train, valid and maybe test DatasetTfm` using `tfms` = (train_tfms,valid_tfms)."
    res = [DatasetTfm(train_ds, tfms[0],  **kwargs),
           DatasetTfm(valid_ds, tfms[1],  **kwargs)]
    if test_ds is not None: res.append(DatasetTfm(test_ds, tfms[1],  **kwargs))
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
        if ds_tfms: datasets = transform_datasets(*datasets, tfms=ds_tfms, size=size, **kwargs)
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
    def from_lists(cls, path:PathOrStr, fnames:FilePathList, labels:Collection[str], valid_pct:int=0.2, test:str=None, **kwargs):
        classes = uniqueify(labels)
        train,valid = random_split(valid_pct, fnames, labels)
        datasets = [ImageClassificationDataset(*train, classes),
                    ImageClassificationDataset(*valid, classes)]
        if test: datasets.append(ImageClassificationDataset.from_single_folder(Path(path)/test, classes=classes))
        return cls.create(*datasets, path=path, **kwargs)

    @classmethod
    def from_name_func(cls, path:PathOrStr, fnames:FilePathList, label_func:Callable, valid_pct:int=0.2, test:str=None, **kwargs):
        labels = [label_func(o) for o in fnames]
        return cls.from_lists(path, fnames, labels, valid_pct=valid_pct, test=test, **kwargs)

    @classmethod
    def from_name_re(cls, path:PathOrStr, fnames:FilePathList, pat:str, valid_pct:int=0.2, test:str=None, **kwargs):
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

    def show_batch(self:DataBunch, rows:int=None, figsize:Tuple[int,int]=(9,10), is_train:bool=True)->None:
        show_image_batch(self.train_dl if is_train else self.valid_dl, self.classes, figsize=figsize, rows=rows)

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

def download_image(url,dest):
    try: r = download_url(url, dest, overwrite=True, show_progress=False)
    except Exception as e: print(f"Error {url} {e}")

def download_images(urls:Collection[str], dest:PathOrStr, max_pics:int=1000, max_workers:int=8):
    "Download images listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_pics]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)

    if max_workers:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(download_image, url, dest/f"{i:08d}.jpg")
                       for i,url in enumerate(urls)]
            for f in progress_bar(as_completed(futures), total=len(urls)): pass
    else:
        for i,url in enumerate(progress_bar(urls)):
            download_image(url, dest/f"{i:08d}.jpg")

def verify_image(file:Path, delete:bool):
    try: assert open_image(file).shape[0]==3
    except Exception as e:
        print(f'{e}')
        if delete: file.unlink()

def verify_images(path:PathOrStr, delete=True, max_workers:int=4):
    "Removes broken images or non 3-channel images in `path`"
    path = Path(path)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        files = list(path.iterdir())
        futures = [ex.submit(verify_image, file, delete=delete) for file in files]
        for f in progress_bar(as_completed(futures), total=len(files)): pass

class ImageFileList(InputList):
    "A list of inputs. Contain methods to get the corresponding labels."
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=image_extensions, recurse=True)->'ImageFileList':
        "Get the list of files in `path` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
        return cls(get_files(path, extensions=extensions, recurse=recurse), path)
