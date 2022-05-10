from fastai.vision.all import *

splitter = GrandparentSplitter(train_name='training', valid_name='testing')
mnist = DataBlock(blocks=(ImageBlock(PILImageBW), CategoryBlock),
                  get_items=get_image_files, splitter=splitter, get_y=parent_label)

if __name__ == '__main__':
    data = mnist.dataloaders(untar_data(URLs.MNIST), bs=256)
    learn = vision_learner(data, resnet18)
    learn.fit_one_cycle(1, 1e-2)

