from fastai.vision.all import *

items = get_image_files(untar_data(URLs.MNIST))
splits = GrandparentSplitter(train_name='training', valid_name='testing')(items)
tds = Datasets(items, [PILImageBW.create, [parent_label, Categorize()]], splits=splits)

if __name__ == '__main__':
    data = tds.dataloaders(bs=256, after_item=[ToTensor(), IntToFloatTensor()]).cuda()
    learn = cnn_learner(data, resnet18)
    learn.fit_one_cycle(1, 1e-2)

