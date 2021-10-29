from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *

path = rank0_first(untar_data, URLs.IMAGEWOOF_320)
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=GrandparentSplitter(valid_name='val'),
    get_items=get_image_files, get_y=parent_label,
    item_tfms=[RandomResizedCrop(160), FlipItem(0.5)],
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders(path, path=path, bs=64)

learn = Learner(dls, xresnet50(n_out=10), metrics=[accuracy,top_k_accuracy]).to_fp16()
with learn.distrib_ctx(): learn.fit_flat_cos(2, 1e-3, cbs=MixUp(0.1))

