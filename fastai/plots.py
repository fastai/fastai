from .imports import *
from .torch_imports import *

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims)
        if (ims.shape[-1] != 3): ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)

    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    """Plots images given image files.

    Arguments:
        im_paths (list): list of paths
        figsize (tuple): figure size
        rows (int): number of rows
        titles (list): list of titles
        maintitle (string): main title
    """
    f = plt.figure(figsize=figsize)
    if maintitle is not None:
        plt.suptitle(maintitle, fontsize=16)    
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, len(imspaths)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)
