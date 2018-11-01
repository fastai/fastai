from torch import topk
from collections import Collection
from ..core import *
from ..basic_train import get_preds
from ..vision.data import DatasetTfm
from ..vision.transform import crop_pad
from pathlib import Path
from ipywidgets import widgets, Layout
from IPython.display import clear_output, HTML

__all__ = ['DatasetFormatter', 'FileDeleter']

# TODO:
# FINISHED button (be done if I dont want to continue)
# Grid 5x5


# Example use: ds, idxs = Converter().from_toplosses(learn.model, data.valid_ds, data.valid_dl, learn.loss_func)
class DatasetFormatter():
    @classmethod
    def from_toplosses(cls, md, ds, dl, loss_func, n_imgs, **kwargs):
        "Formats images with padding for top losses from model `md`, dataset `ds`, `dl`, `loss_func`, with option to limit to `n_imgs` returned."
        dataset, idxs = cls.get_toploss_paths(md, ds, dl, loss_func, n_imgs)
        return cls.padded_ds(dataset, **kwargs), idxs

    @staticmethod
    def get_toploss_paths(md, ds, dl, loss_func, n_imgs=None):
        "Gets filenames for top losses from model `md`, dataset `ds`, `dl`, `loss_func`, with option to limit to `n_imgs` returned."
        if not n_imgs: n_imgs = len(dl)
        _,_,val_losses = get_preds(md, dl, loss_func=loss_func)
        _,idxs = topk(val_losses, n_imgs)
        return ds, idxs

    def padded_ds(ds_input, size=(250, 300), do_crop=False, padding_mode='zeros'):
        "For a Dataset `ds_input`, resize each image in `ds_input` to size `size` by optional cropping (`do_crop`) or padding with `padding_mode`."
        if isinstance(ds_input, DatasetTfm): ds_input = ds_input.ds
        return DatasetTfm(ds_input, crop_pad(), size=size, do_crop=do_crop, padding_mode=padding_mode)


class FileDeleter():
    "Flag images in `file_paths` for deletion and confirm to delete images, showing `batch_size` at a time."

    def __init__(self, dataset, fns_idxs, batch_size:int=5):
        self.all_images,self.batch = [],[]
        self.batch_size = batch_size
        for i in fns_idxs:
            path = Path(dataset.x[i])
            if (path.is_file()):
                img,_ = dataset[i]
                self.all_images.append((img._repr_jpeg_(), path))
                # TODO: SEE IF THIS IS A NEW BOTTLENECK!
        self.render()

    def make_img_widget(self, img, height='250px', width='300px', format='jpg'):
        "Returns an image widget for specified file name."
        return widgets.Image(value=img, format=format, layout=Layout(width=width, height=height))

    def on_confirm(self, btn):
        "Handler for Confirm button click. Deletes all flagged images."
        for img_widget,delete_btn,fp in self.batch:
            fp = delete_btn.file_path
            print('deleting', fp)
            if (delete_btn.flagged_for_delete == True): self.delete_image(fp)
        # Clear current batch from all_imgs
        self.all_images = self.all_images[self.batch_size:]
        self.empty_batch()
        self.render()

    def empty_batch(self): self.batch[:] = []
    def delete_image(self, file_path): os.remove(file_path)
    # TODO: move to .Trash dir

    def on_delete(self, btn):
        "Flags this image as delete or keep."
        btn.button_style = "" if btn.flagged_for_delete else "danger"
        btn.flagged_for_delete = not btn.flagged_for_delete

    def make_button(self, label, file_path=None, handler=None, style=None):
        "Returns a Button widget with specified handler"
        btn = widgets.Button(description=label)
        if handler is not None: btn.on_click(handler)
        if style is not None: btn.button_style = style
        btn.file_path = file_path
        btn.flagged_for_delete = False
        return btn

    def make_horizontal_box(self, children): return widgets.HBox(children)

    def make_vertical_box(self, children, width='auto', height='300px', overflow_x="hidden"):
        return widgets.VBox(children, layout=Layout(width=width, height=height, overflow_x=overflow_x))

    def render(self):
        "Re-renders Jupyter cell for a batch of images."
        clear_output()
        if (len(self.all_images) == 0): return display('No images to show :)')
        widgets_to_render = []
        for img,fp in self.all_images[:self.batch_size]:
            img_widget = self.make_img_widget(img)
            delete_btn = self.make_button('Delete', file_path=fp, handler=self.on_delete)
            widgets_to_render.append(self.make_vertical_box([img_widget, delete_btn]))
            self.batch.append((img_widget, delete_btn, fp))
        display(self.make_horizontal_box(widgets_to_render))
        display(self.make_button('Confirm', handler=self.on_confirm, style="primary"))

    # Initial implementation by:
    # Zach Caceres @zachcaceres (https://github.com/zcaceres)
    # Jason Patnick (https://github.com/pattyhendrix)
    # Francisco Ingham @inghamfran (https://github.com/lesscomfortable)
