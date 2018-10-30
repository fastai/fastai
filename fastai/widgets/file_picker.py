from ipywidgets import widgets, Layout
from IPython.display import clear_output, HTML
from ..fastai.core import *

__all__ = ['file_deleter']

# TODO:
# FINISHED button (be done if I dont want to continue)
# Grid 5x5

def FileDeleter():
    "Flag images in `file_paths` for deletion and confirm to delete images, showing `batch_size` at a time."

    def __init__(self, file_paths:Collection[PathOrStr], batch_size:int=5):
        self.all_images,self.batch = [],[]
        for fp in [o for o in map(Path,file_paths) if o.is_file()]:
            img = self.make_img(fp)
            delete_btn = self.make_button('Delete', file_path=fp, handler=on_delete)
            self.all_images.append((img, delete_btn, fp))
        render()

    def make_img(self, file_path, height='250px', width='300px', format='jpg'):
        "Returns an image widget for specified file name."
        with open(file_path, 'rb') as opened:
            read_file = opened.read()
            return widgets.Image(value=read_file, format=format, layout=Layout(width=width, height=height))

    def on_confirm(self, btn):
        "Handler for Confirm button click. Deletes all flagged images."
        to_remove = []
        for img,delete_btn,fp in self.batch:
            fp = delete_btn.file_path
            if (delete_btn.flagged_for_delete == True): delete_image(fp)
            to_remove.append((img, delete_btn, fp))
        for img, delete_btn, fp in to_remove:
            self.all_images.remove((img, delete_btn, fp))
        empty_batch()
        render()

    def empty_batch(self): self.batch[:] = []
    def delete_image(self, file_path): os.remove(file_path)

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

    def make_vertical_box(self, children, width='auto', height='300px'):
        return widgets.VBox(children, layout=Layout(width=width, height=height))

    def render(self):
        "Re-renders Jupyter cell for a batch of images."
        clear_output()
        if (len(self.all_images) == 0): return display('No images to show :)')
        widgets_to_render = []
        for img, delete_btn, fp in self.all_images[:self.batch_size]:
            widgets_to_render.append(make_vertical_box([img, delete_btn]))
            self.batch.append((img, delete_btn, fp))
        display(make_horizontal_box(widgets_to_render))
        display(make_button('Confirm', handler=on_confirm, style="primary"))

    # Initial implementation by:
    # Zach Caceres @zachcaceres (https://github.com/zcaceres)
    # Jason Patnick (https://github.com/pattyhendrix)
    # Francisco Ingham @inghamfran (https://github.com/lesscomfortable)

