from ..torch_core import *
from ..basic_data import *
from ..basic_train import *
from .image import *
from ..train import Interpretation

__all__ = ['SegmentationInterpretation', 'ObjectDetectionInterpretation', 'MultiLabelClassificationInterpretation']

class SegmentationInterpretation(Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor,
                 ds_type:DatasetType=DatasetType.Valid):
        super(SegmentationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        self.pred_class = self.preds.argmax(dim=1)
        self.c2i = {c:i for i,c in enumerate(self.data.classes)}
        self.i2c = {i:c for c,i in self.c2i.items()}
    
    def top_losses(self, sizes:Tuple, k:int=None, largest=True):
        "Reduce flatten loss to give a single loss value for each image"
        losses = self.losses.view(-1, np.prod(sizes)).mean(-1)
        return losses.topk(ifnone(k, len(losses)), largest=largest)
    
    def _interp_show(self, ims:ImageSegment, classes:Collection, sz:int=20, cmap='tab20',
                    title_suffix:str=None):
        fig,axes=plt.subplots(1,2,figsize=(sz,sz))
        class_idxs = [self.c2i[c] for c in classes]
        
        #image
        mask = torch.cat([ims.data==i for i in class_idxs]).max(dim=0)[0][None,:].long()
        masked_im = image2np(ims.data*mask)
        im=axes[0].imshow(masked_im, cmap=cmap)

        #labels
        masked_im_labels = list(np.unique(masked_im))
        c = len(masked_im_labels); n = math.ceil(np.sqrt(c))
        label_im = np.array(masked_im_labels + [np.nan]*(n**2-c)).reshape(n,n)
        axes[1].imshow(label_im, cmap=cmap)
        for i,l in enumerate([self.i2c[l] for l in masked_im_labels]):
            div,mod=divmod(i,n)
            axes[1].text(mod, div, f"{l}", ha='center', color='white', fontdict={'size':sz})

        if title_suffix:
            axes[0].set_title(f"{title_suffix}_imsegment")
            axes[1].set_title(f"{title_suffix}_labels")

    def show_xyz(self, i, classes=None, sz=10):
        'show (image, true and pred) from dataset with color mappings'
        classes = ifnone(classes, self.data.classes)
        x,y = self.ds[i]
        self.data.valid_ds.x.show_xys([x],[y], figsize=(sz/2,sz/2))
        self._interp_show(ImageSegment(self.y_true[i]), classes, sz=sz, title_suffix='true')
        self._interp_show(ImageSegment(self.pred_class[i][None,:]), classes, sz=sz, title_suffix='pred')


class ObjectDetectionInterpretation(Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid):
        raise NotImplementedError
        super(ObjectDetectionInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        

class MultiLabelClassificationInterpretation(Interpretation):
    "Interpretation methods for classification models."
    def __init__(self, learn:Learner, preds:Tensor, y_true:Tensor, losses:Tensor, ds_type:DatasetType=DatasetType.Valid,
                     sigmoid:bool=True, thresh:float=0.3):
        raise NotImplementedError
        super(MultiLabelClassificationInterpretation, self).__init__(learn,preds,y_true,losses,ds_type)
        self.pred_class = self.preds.sigmoid(dim=1)>thresh if sigmoid else self.preds>thresh