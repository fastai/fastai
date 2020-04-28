import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.vision import *
from fastai.vision.data import verify_image
from utils.text import *
import PIL

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

@pytest.fixture(scope="module")
def path_var_size():
    path = untar_data(URLs.MNIST_VAR_SIZE_TINY)
    return path

def mnist_tiny_sanity_test(data):
    assert data.c == 2
    assert set(map(str, set(data.classes))) == {'3', '7'}

def test_path_can_be_str_type(path):
    this_tests(ImageDataBunch.from_csv)
    assert ImageDataBunch.from_csv(str(path))

def test_from_folder(path):
    this_tests(ImageDataBunch.from_folder)
    for valid_pct in [None, 0.9]:
        data = ImageDataBunch.from_folder(path, test='test', valid_pct=valid_pct)
        mnist_tiny_sanity_test(data)
        if valid_pct:
            n_valid = len(data.valid_ds)
            n_train = len(data.train_ds)
            n_total = n_valid + n_train
            assert n_valid == int(n_total * valid_pct)

def test_from_name_re(path):
    this_tests(ImageDataBunch.from_name_re)
    fnames = get_image_files(path/'train', recurse=True)
    pat = r'/([^/]+)\/\d+.png$'
    data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=(rand_pad(2, 28), []))
    mnist_tiny_sanity_test(data)

def test_from_lists(path):
    this_tests(ImageDataBunch.from_lists)
    df = pd.read_csv(path/'labels.csv')
    fnames = [path/f for f in df['name'].values]
    labels = df['label'].values
    data = ImageDataBunch.from_lists(path, fnames, labels)
    mnist_tiny_sanity_test(data)
    #Check labels weren't shuffled for the validation set
    valid_fnames = data.valid_ds.x.items
    pat = re.compile(r'/([^/]+)/\d+.png$')
    expected_labels = [int(pat.search(str(o)).group(1)) for o in valid_fnames]
    current_labels = [int(str(l)) for l in data.valid_ds.y]
    assert len(expected_labels) == len(current_labels)
    assert np.all(np.array(expected_labels) == np.array(current_labels))

def test_from_csv_and_from_df(path):
    this_tests(ImageDataBunch.from_csv, ImageDataBunch.from_df)
    for func in ['from_csv', 'from_df']:
        files = []
        if func == 'from_df': data = ImageDataBunch.from_df(path, df=pd.read_csv(path/'labels.csv'), size=28)
        else: data = ImageDataBunch.from_csv(path, size=28)
        mnist_tiny_sanity_test(data)

rms = ['PAD', 'CROP', 'SQUISH']

def check_resized(data, size, args):
    x,_ = data.train_ds[0]
    size_want = (size, size) if isinstance(size, int) else size
    size_real = x.size
    assert size_want == size_real, f"[{args}]: size mismatch after resize {size} expected {size_want}, got {size_real}"

def test_image_resize(path, path_var_size):
    this_tests(ImageDataBunch.from_name_re)
    # in this test the 2 datasets are:
    # (1) 28x28,
    # (2) var-size but larger than 28x28,
    # and the resizes are always less than 28x28, so it always tests a real resize
    for p in [path, path_var_size]: # identical + var sized inputs
        fnames = get_image_files(p/'train', recurse=True)
        pat = r'/([^/]+)\/\d+.png$'
        for size in [14, (14,14), (14,20)]:
            for rm_name in rms:
                rm = getattr(ResizeMethod, rm_name)
                args = f"path={p}, size={size}, resize_method={rm_name}"

                # resize the factory method way
                with CaptureStderr() as cs:
                    data = ImageDataBunch.from_name_re(p, fnames, pat, ds_tfms=None, size=size, resize_method=rm)
                assert len(cs.err)==0, f"[{args}]: got collate_fn warning {cs.err}"
                check_resized(data, size, args)

                # resize the data block way
                with CaptureStderr() as cs:
                    data = (ImageList.from_folder(p)
                            .split_none()
                            .label_from_folder()
                            .transform(size=size, resize_method=rm)
                            .databunch(bs=2)
                            )
                assert len(cs.err)==0, f"[{args}]: got collate_fn warning {cs.err}"
                check_resized(data, size, args)

def test_multi_iter_broken(path):
    this_tests('na')
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    for i in range(2): x,y = next(iter(data.train_dl))

def test_multi_iter(path):
    this_tests('na')
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    for i in range(2): x,y = data.one_batch()

def test_clean_tear_down(path):
    this_tests('na')
    docstr = "test DataLoader iter doesn't get stuck"
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    data.normalize()

def test_normalize(path):
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    x,y = data.one_batch(ds_type=DatasetType.Valid, denorm=False)
    m,s = x.mean(),x.std()
    this_tests(data.normalize)
    data.normalize()
    x,y = data.one_batch(ds_type=DatasetType.Valid, denorm=False)
    assert abs(x.mean()) < abs(m)
    assert abs(x.std()-1) < abs(m-1)

    with pytest.raises(Exception): data.normalize()
    data.valid_dl = None
    with pytest.raises(Exception): data.normalize()

def test_denormalize(path):
    this_tests(denormalize)
    data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []))
    original_x, y = data.one_batch(ds_type=DatasetType.Valid, denorm=False)
    data.normalize()
    normalized_x, y = data.one_batch(ds_type=DatasetType.Valid, denorm=False)
    denormalized = denormalize(normalized_x, *data.stats)
    assert round(original_x.mean().item(), 3) == round(denormalized.mean().item(), 3)
    assert round(original_x.std().item(), 3) == round(denormalized.std().item(), 3)

def test_download_images():
    this_tests(download_images)
    base_url = 'http://files.fast.ai/data/tst_images/'
    fnames = ['tst0.jpg', 'tst1.png', 'tst2.tif']

    tmp_path = URLs.LOCAL_PATH/'data'/'tmp'
    try:
        os.makedirs(tmp_path)
        with open(tmp_path/'imgs.txt', 'w') as f:
            [f.write(f'{base_url}{fname}\n') for fname in fnames]
        download_images(tmp_path/'imgs.txt', tmp_path)
        for fname in fnames:
            ext = fname.split('.')[-1]
            files = list(tmp_path.glob(f'*.{ext}'))
            assert len(files) == 1
            assert os.path.getsize(files[0]) > 0
    finally:
        shutil.rmtree(tmp_path)

responses = try_import('responses')
@pytest.mark.skipif(not responses, reason="requires the `responses` module")
def test_trunc_download():
    this_tests(untar_data)
    url = URLs.COCO_TINY
    fname = datapath4file(url2name(url)).with_suffix(".tgz")
    # backup user's current state
    fname_bak = fname.parent/f"{fname.name}-bak"
    if fname.exists(): os.rename(fname, fname_bak)

    with responses.RequestsMock() as rsps:
        mock_headers = {'Content-Type':'text/plain', 'Content-Length':'168168549'}
        rsps.add(responses.GET, f"{url}.tgz",
                 body="some truncated text", status=200, headers=mock_headers)
        try: coco = untar_data(url, force_download=True)
        except AssertionError as e:
            expected_error = f"Downloaded file {fname} does not match checksum expected! Remove that file from {Config().data_path()} and try your code again."
            assert e.args[0] == expected_error
        except:
            assert False, f"untar_data({URLs.COCO_TINY}) had Unexpected error: {sys.exc_info()[0]}"
        else:
            assert False, f"untar_data({URLs.COCO_TINY})  should have gracefully failed on a truncated download"
        finally:
            # restore user's original state
            if fname.exists():     os.remove(fname)
            if fname_bak.exists(): os.rename(fname_bak, fname)

def test_verify_images(path):
    this_tests(verify_images)
    tmp_path = path/'tmp'
    os.makedirs(tmp_path, exist_ok=True)
    verify_images(path/'train'/'3', dest=tmp_path, max_size=27, max_workers=4)
    images = list(tmp_path.iterdir())
    assert len(images) == 346
    img = PIL.Image.open(images[0])
    assert img.height == 27 and img.width == 27
    shutil.rmtree(tmp_path)

def test_verify_image(path):
    this_tests(verify_image)
    tmp_path = path/'tmp'
    os.makedirs(tmp_path, exist_ok=True)
    verify_image(path/'train'/'3'/'867.png', 0, False, dest=tmp_path, max_size=27)
    img = PIL.Image.open(tmp_path/'867.png')
    assert img.height == 27 and img.width == 27
    shutil.rmtree(tmp_path)

#Data block
def _print_data(data): print(len(data.train_ds),len(data.valid_ds))
def _check_data(data, t, v):
    assert len(data.train_ds)==t
    assert len(data.valid_ds)==v
    _ = data.train_ds[0]

def test_vision_datasets():
    this_tests(ImageList.from_folder)
    il = ImageList.from_folder(untar_data(URLs.MNIST_TINY))
    sds = il.split_by_idx([0]).label_from_folder().add_test_folder()
    assert np.array_equal(sds.train.classes, sds.valid.classes), 'train/valid classes same'
    assert len(sds.test)==20, "test_ds is correct size"
    this_tests(sds.databunch)
    data = sds.databunch()
    _check_data(data, len(il)-1, 1)

def test_multi():
    this_tests(ImageList.from_csv)
    path = untar_data(URLs.PLANET_TINY)
    data = (ImageList.from_csv(path, 'labels.csv', folder='train', suffix='.jpg')
            .split_by_rand_pct(seed=42).label_from_df(label_delim=' ').databunch())
    x,y = data.valid_ds[0]
    assert x.shape[0]==3
    assert data.c==len(y.data)==14
    assert len(str(y))>2
    _check_data(data, 160, 40)

def test_camvid():
    this_tests(SegmentationItemList)
    camvid = untar_data(URLs.CAMVID_TINY)
    path_lbl = camvid/'labels'
    path_img = camvid/'images'
    codes = np.loadtxt(camvid/'codes.txt', dtype=str)
    get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
    data = (SegmentationItemList.from_folder(path_img)
            .split_by_rand_pct()
            .label_from_func(get_y_fn, classes=codes)
            .transform(get_transforms(), tfm_y=True)
            .databunch())
    _check_data(data, 80, 20)

def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)

def test_points():
    this_tests(PointsItemList)
    coco = untar_data(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(coco/'train.json')
    points = [tensor([b[0][0][0], b[0][0][1]]) for b in lbl_bbox]
    img2pnts = dict(zip(images, points))
    get_y_func = lambda o:img2pnts[o.name]
    data = (PointsItemList.from_folder(coco)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .databunch())
    _check_data(data,160,40)

def test_coco():
    this_tests(ObjectItemList)
    coco = untar_data(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(coco/'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o:img2bbox[o.name]
    data = (ObjectItemList.from_folder(coco)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=16, collate_fn=bb_pad_collate))
    _check_data(data, 160, 40)

def test_coco_same_size():
    this_tests(ObjectItemList)
    def get_y_func(fname):
        cat = fname.parent.name
        bbox = torch.cat([torch.randint(0,5,(2,)), torch.randint(23,28,(2,))])
        bbox = list(bbox.float().numpy())
        return [[bbox, bbox], [cat, cat]]

    coco = untar_data(URLs.MNIST_TINY)
    bs = 16
    data = (ObjectItemList.from_folder(coco)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=16, collate_fn=bb_pad_collate))
    _check_data(data, 1143, 285)

def test_coco_pickle():
    this_tests(ObjectItemList)
    coco = untar_data(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(coco/'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o:img2bbox[o.name]
    tfms = get_transforms()
    pickle_tfms = pickle.dumps(tfms)
    unpickle_tfms = pickle.loads(pickle_tfms)
    data = (ObjectItemList.from_folder(coco)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(unpickle_tfms, tfm_y=True)
            .databunch(bs=16, collate_fn=bb_pad_collate))
    _check_data(data, 160, 40)

def test_image_to_image_different_y_size():
    this_tests(get_transforms)
    get_y_func = lambda o:o
    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms()
    data = (ImageImageList.from_folder(mnist)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(tfms, size=20)
            .transform_y(size=80)
            .databunch(bs=16))

    x,y = data.one_batch()
    assert x.shape[2]*4 == y.shape[3]

def test_image_to_image_different_tfms():
    this_tests(get_transforms)
    get_y_func = lambda o:o
    mnist = untar_data(URLs.COCO_TINY)
    x_tfms = get_transforms()
    y_tfms = [[t for t in x_tfms[0]], [t for t in x_tfms[1]]]
    y_tfms[0].append(flip_lr())
    data = (ImageImageList.from_folder(mnist)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(x_tfms)
            .transform_y(y_tfms)
            .databunch(bs=16))

    x,y = data.one_batch()
    x1 = x[0]
    y1 = y[0]
    x1r = flip_lr(Image(x1)).data
    assert (y1 == x1r).all()

def test_vision_pil2tensor():
    this_tests(pil2tensor)
    path  = Path(__file__).parent / "data/test/images"
    files = list(Path(path).glob("**/*.*"))
    pil_passed, pil_failed = [],[]
    for f in files:
        try:
            im = PIL.Image.open(f)
            #provoke read of the file so we can isolate PIL issue separately
            b = np.asarray(im.convert("RGB"))
            pil_passed.append(f)
        except:
            pil_failed.append(f)

    pil2tensor_passed,pil2tensor_failed = [],[]
    for f in pil_passed:
        try :
            # it doesn't matter for the test if we convert "RGB" or "I"
            im = PIL.Image.open(f).convert("RGB")
            t  = pil2tensor(im,np.float)
            pil2tensor_passed.append(f)
        except:
            pil2tensor_failed.append(f)
            print(f"converting file: {f}  had Unexpected error:", sys.exc_info()[0])

    if len(pil2tensor_failed)>0 :
        print("\npil2tensor failed to convert the following images:")
        [print(f) for f in pil2tensor_failed]

    assert(len(pil2tensor_passed) == len(pil_passed))

def test_vision_pil2tensor_16bit():
    this_tests(pil2tensor)
    f    = Path(__file__) .parent/ "data/test/images/gray_16bit.png"
    im   = PIL.Image.open(f).convert("I") # so that the 16bit values are preserved as integers
    vmax = pil2tensor(im,np.int).data.numpy().max()
    assert(vmax>255)

def test_vision_pil2tensor_numpy():
    this_tests(pil2tensor)
    "assert that the two arrays contains the same values"
    arr  = np.random.rand(16,16,3)
    diff = np.sort( pil2tensor(arr,np.float).data.numpy().flatten() ) - np.sort(arr.flatten())
    assert( np.sum(diff==0)==len(arr.flatten()) )
