from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from ..vision.data import *
from ..vision.transform import *
from ..vision.image import open_image
from ipywidgets import widgets, Layout
from IPython.display import clear_output, HTML

__all__ = ['DatasetFormatter', 'ImageRelabeler']

# Example use: ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)
# ImageRelabeler(ds, idxs)
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
    def __init__(self, dataset, fns_idxs, batch_size:int=5):
        self._all_images,self._batch = [],[]
        self._batch_size = batch_size
        self._labels = dataset.classes
        self._all_images = [(dataset.x[i]._repr_jpeg_(), dataset.x.items[i], self._labels[dataset.y[i].data])
                            for i in fns_idxs if dataset.x.items[i].is_file()]

    def empty_batch(self): self._batch[:] = []

    @classmethod
    def make_img_widget(cls, img, height='250px', width='300px', format='jpg'):
        "Returns an image widget for specified file name."
        return widgets.Image(value=img, format=format, layout=Layout(width=width, height=height))

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
    def make_dropdown_widget(cls, description='Description', options=['Label 1', 'Label 2'], value='Label 1', file_path=None, layout=Layout(), handler=None):
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        if file_path is not None: dd.file_path = file_path
        if handler is not None: dd.observe(handler, names=['value'])
        return dd

    @classmethod
    def make_horizontal_box(cls, children): return widgets.HBox(children)

    @classmethod
    def make_vertical_box(cls, children, width='auto', height='300px', overflow_x="hidden"):
        return widgets.VBox(children, layout=Layout(width=width, height=height, overflow_x=overflow_x))

class ImageRelabeler(ImageCleaner):
    "Displays images with their current label and, if labeled incorrectly, allows user to move image to properly labeled folder."
    def __init__(self, dataset, fns_idxs, batch_size:int=5):
        super().__init__(dataset, fns_idxs, batch_size=batch_size)
        self.render()

    def relabel(self, change):
        "Relabel images by moving from parent dir with old label `class_old` to parent dir with new label `class_new`"
        class_new,class_old,file_path = change.new,change.old,change.owner.file_path
        fp = Path(file_path)
        parent = fp.parents[1]
        # TODO: disambiguate relabeling process based on label type (CSV, folders etc.)
        new_filepath = Path(f'{parent}/{class_new}/{fp.name}')
        if new_filepath.exists():
            new_filepath = Path(f'{parent}/{class_new}/{fp.stem}-moved{fp.suffix}')
        Path(file_path).replace(new_filepath)
        change.owner.file_path = new_filepath

    def next_batch(self, btn):
        self._all_images = self._all_images[self._batch_size:]
        self.empty_batch()
        self.render()

    # TODO: refactor some of this out to parent
    def render(self):
        "Re-render Jupyter cell for batch of images"
        clear_output()
        if (len(self._all_images) == 0): return display('No images to show :)')
        widgets_to_render = []
        for (img,fp,human_readable_label) in self._all_images[:self._batch_size]:
            img_widget = self.make_img_widget(img)
            dropdown = self.make_dropdown_widget(description='', options=self._labels, value=human_readable_label, file_path=fp, handler=self.relabel, layout=Layout(width='auto'))
            delete_btn = self.make_button_widget('Delete', file_path=fp, handler=self.on_delete)
            widgets_to_render.append(self.make_vertical_box([img_widget, dropdown, delete_btn], height='300px'))
            self._batch.append((img_widget, dropdown, fp, human_readable_label))
        display(self.make_horizontal_box(widgets_to_render))
        display(self.make_button_widget('Next Batch', handler=self.next_batch, style="primary"))
