from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from ..vision.data import *
from ..vision.transform import *
from ..vision.image import open_image
from ipywidgets import widgets, Layout
from IPython.display import clear_output, HTML

__all__ = ['DatasetFormatter', 'ImageCleaner']

class DatasetFormatter():
    @classmethod
    def from_toplosses(cls, learn, n_imgs=None, ds_type:DatasetType=DatasetType.Valid, **kwargs):
        "Formats images with padding for top losses from learner `learn`, using dataset type `ds_type`, with option to limit to `n_imgs` returned."
        dl = learn.dl(ds_type)
        if not n_imgs: n_imgs = len(dl.dataset)
        _,_,val_losses = learn.get_preds(ds_type, with_loss=True)
        idxs = torch.topk(val_losses, n_imgs)[1]
        return cls.padded_ds(dl.dataset, **kwargs), idxs

    def padded_ds(ll_input, size=(250, 300), do_crop=False, padding_mode='zeros'):
        "For a LabelList `ll_input`, resize each image in `ll_input` to size `size` by optional cropping (`do_crop`) or padding with `padding_mode`."
        return ll_input.transform(crop_pad(), size=size, do_crop=do_crop, padding_mode=padding_mode)

class ImageCleaner():
    "Displays images with their current label. If image is junk data or labeled incorrectly, allows user to delete image or move image to properly labeled folder."
    def __init__(self, dataset, fns_idxs, batch_size:int=5):
        self._all_images,self._batch = [],[]
        self._batch_size = batch_size
        self._labels = dataset.classes
        self._all_images = [(dataset.x[i]._repr_jpeg_(), dataset.x.items[i], self._labels[dataset.y[i].data])
                            for i in fns_idxs if dataset.x.items[i].is_file()]
        self.render()

    @classmethod
    def make_img_widget(cls, img, layout=Layout(), format='jpg'):
        "Returns an image widget for specified file name."
        return widgets.Image(value=img, format=format, layout=layout)

    @classmethod
    def make_button_widget(cls, label, file_path=None, handler=None, style=None, layout=Layout(width='auto')):
        "Returns a Button widget with specified handler"
        btn = widgets.Button(description=label, layout=layout)
        if handler is not None: btn.on_click(handler)
        if style is not None: btn.button_style = style
        btn.file_path = file_path
        btn.flagged_for_delete = False
        return btn

    @classmethod
    def make_dropdown_widget(cls, description='Description', options=['Label 1', 'Label 2'], value='Label 1',
                            file_path=None, layout=Layout(), handler=None):
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        if file_path is not None: dd.file_path = file_path
        if handler is not None: dd.observe(handler, names=['value'])
        return dd

    @classmethod
    def make_horizontal_box(cls, children, layout=Layout()): return widgets.HBox(children, layout=layout)

    @classmethod
    def make_vertical_box(cls, children, layout=Layout()): return widgets.VBox(children, layout=layout)

    def relabel(self, change):
        "Relabel images by moving from parent dir with old label `class_old` to parent dir with new label `class_new`"
        class_new,class_old,file_path = change.new,change.old,change.owner.file_path
        fp = Path(file_path)
        parent = fp.parents[1]
        # TODO: disambiguate relabeling process based on label type (CSV, folders etc.)
        new_filepath = Path(f'{parent}/{class_new}/{fp.name}')
        if new_filepath.exists():
            new_filepath = Path(f'{parent}/{class_new}/{fp.stem}-moved{fp.suffix}')
        fp.replace(new_filepath)
        change.owner.file_path = new_filepath

    def next_batch(self, btn):
        "Handler for 'Next Batch' button click. Deletes all flagged images and renders next batch."
        for img_widget, delete_btn, fp, in self._batch:
            fp = delete_btn.file_path
            if (delete_btn.flagged_for_delete == True): self.delete_image(fp)
        self._all_images = self._all_images[self._batch_size:]
        self.empty_batch()
        self.render()

    def on_delete(self, btn):
        "Flags this image as delete or keep."
        btn.button_style = "" if btn.flagged_for_delete else "danger"
        btn.flagged_for_delete = not btn.flagged_for_delete

    def empty_batch(self): self._batch[:] = []

    def delete_image(self, file_path): os.remove(file_path)
    # TODO: move to .Trash dir

    def empty(self): return len(self._all_images) == 0

    def get_widgets(self):
        "Create and format widget set"
        widgets = []
        for (img,fp,human_readable_label) in self._all_images[:self._batch_size]:
            img_widget = self.make_img_widget(img, layout=Layout(height='250px', width='300px'))
            dropdown = self.make_dropdown_widget(description='', options=self._labels, value=human_readable_label,
                                                 file_path=fp, handler=self.relabel, layout=Layout(width='auto'))
            delete_btn = self.make_button_widget('Delete', file_path=fp, handler=self.on_delete)
            widgets.append(self.make_vertical_box([img_widget, dropdown, delete_btn],
                                                  layout=Layout(width='auto', height='300px', overflow_x="hidden")))
            self._batch.append((img_widget, delete_btn, fp))
        return widgets

    def render(self):
        "Re-render Jupyter cell for batch of images"
        clear_output()
        if (self.empty()): return display('No images to show :)')
        display(self.make_horizontal_box(self.get_widgets()))
        display(self.make_button_widget('Next Batch', handler=self.next_batch, style="primary"))
