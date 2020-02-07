from abc import ABC
from itertools import chain, islice
from math import ceil

from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from ..data_block import LabelLists
from ..vision.transform import *
from ..vision.image import *
from ..callbacks.hooks import *
from ..layers import *
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display

__all__ = ['DatasetFormatter', 'ImageCleaner', 'PredictionsCorrector', 'data_deleter']

class DatasetFormatter():
    "Returns a dataset with the appropriate format and file indices to be displayed."
    @classmethod
    def from_toplosses(cls, learn, n_imgs=None, **kwargs):
        "Gets indices with top losses."
        train_ds, train_idxs = cls.get_toplosses_idxs(learn, n_imgs, **kwargs)
        return train_ds, train_idxs

    @classmethod
    def get_toplosses_idxs(cls, learn, n_imgs, **kwargs):
        "Sorts `ds_type` dataset by top losses and returns dataset and sorted indices."
        dl = learn.data.fix_dl
        if not n_imgs: n_imgs = len(dl.dataset)
        _,_,top_losses = learn.get_preds(ds_type=DatasetType.Fix, with_loss=True)
        idxs = torch.topk(top_losses, n_imgs)[1]
        return cls.padded_ds(dl.dataset, **kwargs), idxs

    @staticmethod
    def padded_ds(ll_input, size=(250, 300), resize_method=ResizeMethod.CROP, padding_mode='zeros', **kwargs):
        "For a LabelList `ll_input`, resize each image to `size` using `resize_method` and `padding_mode`."
        return ll_input.transform(tfms=crop_pad(), size=size, resize_method=resize_method, padding_mode=padding_mode)

    @classmethod
    def from_similars(cls, learn, layer_ls:list=[0, 7, 2], **kwargs):
        "Gets the indices for the most similar images."
        train_ds, train_idxs = cls.get_similars_idxs(learn, layer_ls, **kwargs)
        return train_ds, train_idxs

    @classmethod
    def get_similars_idxs(cls, learn, layer_ls, **kwargs):
        "Gets the indices for the most similar images in `ds_type` dataset"
        hook = hook_output(learn.model[layer_ls[0]][layer_ls[1]][layer_ls[2]])
        dl = learn.data.fix_dl

        ds_actns = cls.get_actns(learn, hook=hook, dl=dl, **kwargs)
        similarities = cls.comb_similarity(ds_actns, ds_actns, **kwargs)
        idxs = cls.sort_idxs(similarities)
        return cls.padded_ds(dl, **kwargs), idxs

    @staticmethod
    def get_actns(learn, hook:Hook, dl:DataLoader, pool=AdaptiveConcatPool2d, pool_dim:int=4, **kwargs):
        "Gets activations at the layer specified by `hook`, applies `pool` of dim `pool_dim` and concatenates"
        print('Getting activations...')

        actns = []
        learn.model.eval()
        with torch.no_grad():
            for (xb,yb) in progress_bar(dl):
                learn.model(xb)
                actns.append((hook.stored).cpu())

        if pool:
            pool = pool(pool_dim)
            return pool(torch.cat(actns)).view(len(dl.x),-1)
        else: return torch.cat(actns).view(len(dl.x),-1)


    @staticmethod
    def comb_similarity(t1: torch.Tensor, t2: torch.Tensor, **kwargs):
        # https://github.com/pytorch/pytorch/issues/11202
        "Computes the similarity function between each embedding of `t1` and `t2` matrices."
        print('Computing similarities...')

        w1 = t1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if t2 is t1 else t2.norm(p=2, dim=1, keepdim=True)

        t = torch.mm(t1, t2.t()) / (w1 * w2.t()).clamp(min=1e-8)
        return torch.tril(t, diagonal=-1)

    def largest_indices(arr, n):
        "Returns the `n` largest indices from a numpy array `arr`."
        #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)

    @classmethod
    def sort_idxs(cls, similarities):
        "Sorts `similarities` and return the indexes in pairs ordered by highest similarity."
        idxs = cls.largest_indices(similarities, len(similarities))
        idxs = [(idxs[0][i], idxs[1][i]) for i in range(len(idxs[0]))]
        return [e for l in idxs for e in l]

    @classmethod
    def from_most_unsure(cls, learn:Learner, num=50) -> Tuple[DataLoader, List[int], Sequence[str], List[str]]:
        """
        Gets `num` items from the test set, for which the difference in probabilities between
        the most probable and second most probable classes is minimal.
        """
        preds, _ = learn.get_preds(DatasetType.Test)
        classes = learn.data.train_dl.classes
        labels = [classes[i] for i in preds.argmax(dim=1)]

        most_unsure = preds.topk(2, dim=1)[0] @ torch.tensor([1.0, -1.0])
        most_unsure.abs_()
        idxs = most_unsure.argsort()[:num].tolist()
        return cls.padded_ds(learn.data.test_dl), idxs, classes, labels

@dataclass
class ImgData:
    jpg_blob: bytes
    label: str
    payload: Mapping

class BasicImageWidget(ABC):
    def __init__(self, dataset:LabelLists, fns_idxs:Collection[int], batch_size=5, drop_batch_on_nonfile=False,
                 classes:Optional[Sequence[str]]=None, labels:Optional[Sequence[str]]=None,
                 before_next_batch:Optional[Callable[[Tuple[Mapping, ...]], Any]]=None):
        super().__init__()
        self._dataset,self.batch_size,self._labels,self.before_next_batch = dataset,batch_size,labels,before_next_batch
        self._classes = classes or dataset.classes
        self._all_images = self.create_image_list(fns_idxs, drop_batch_on_nonfile)

    @staticmethod
    def make_img_widget(img:bytes, layout=Layout(height='250px', width='300px'), format='jpg') -> widgets.Image:
        "Returns an image widget for specified file name `img`."
        return widgets.Image(value=img, format=format, layout=layout)

    @staticmethod
    def make_button_widget(label:str, handler:Callable, img_idx:Optional[int]=None,
                           style:str=None, layout=Layout(width='auto')) -> widgets.Button:
        "Return a Button widget with specified `handler`."
        btn = widgets.Button(description=label, layout=layout)
        btn.on_click(handler)
        if style is not None: btn.button_style = style
        if img_idx is not None: btn.img_idx = img_idx
        return btn

    @staticmethod
    def make_dropdown_widget(options:Collection, value, handler:Callable, img_idx:Optional[int]=None,
                             description='', layout=Layout(width='auto')) -> widgets.Dropdown:
        "Return a Dropdown widget with specified `handler`."
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        dd.observe(handler, names='value')
        if img_idx is not None: dd.img_idx = img_idx
        return dd

    @staticmethod
    def make_horizontal_box(children:Collection[widgets.Widget], layout=Layout()) -> widgets.HBox:
        "Make a horizontal box with `children` and `layout`."
        return widgets.HBox(children, layout=layout)

    @staticmethod
    def make_vertical_box(children:Collection[widgets.Widget],
                          layout=Layout(width='auto', height='300px', overflow_x="hidden")) -> widgets.VBox:
        "Make a vertical box with `children` and `layout`."
        return widgets.VBox(children, layout=layout)

    def create_image_list(self, fns_idxs:Collection[int], drop_batch_on_nonfile=False) -> Iterator[ImgData]:
        "Create a list of images, filenames and labels but first removing files that are not supposed to be displayed."
        items = self._dataset.x.items
        idxs = ((i for i in fns_idxs if Path(items[i]).is_file())
                if not drop_batch_on_nonfile
                else chain.from_iterable(c for c in chunks(fns_idxs, self.batch_size)
                                           if all(Path(items[i]).is_file() for i in c)))
        for i in idxs: yield ImgData(self._dataset.x[i]._repr_jpeg_(), self._get_label(i), self.make_payload(i))

    def _get_label(self, idx):
        "Returns a label for an image with the given `idx`."
        return self._labels[idx] if self._labels is not None else self._classes[self._dataset.y[idx].data]

    @abstractmethod
    def make_payload(self, idx:int) -> Mapping:
        "Override in a subclass to associate an image with the given `idx` with a custom payload."
        pass

    def _get_change_payload(self, change_owner):
        """
        Call in widget's on change handler to retrieve the payload.
        Assumes the widget was created by a factory method taking `img_idx` parameter.
        """
        return self._batch_payloads[change_owner.img_idx]

    def next_batch(self, _=None):
        "Fetches a next batch of images for rendering."
        if self.before_next_batch and hasattr(self, '_batch_payloads'): self.before_next_batch(self._batch_payloads)
        batch = tuple(islice(self._all_images, self.batch_size))
        self._batch_payloads = tuple(b.payload for b in batch)
        self.render(batch)

    @abstractmethod
    def render(self, batch:Tuple[ImgData]):
        "Override in a subclass to render the widgets for a batch of images."
        pass

def data_deleter(path:PathOrStr, dataset:LabelLists, del_idx:Collection[int]):
    "Delete the data you want by index.Save changes in path as 'cleaned.csv'."
    csv_dict = {dataset.x.items[i]: dataset.y[i] for i in range(len(dataset))}
    for del_path in dataset.x.items[del_idx]:
        del csv_dict[del_path]
    csv_path = Path(path) / 'cleaned.csv'
    with open(csv_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['name', 'label'])
        for pair in csv_dict.items():
            pair = [os.path.relpath(pair[0], path), pair[1]]
            csv_writer.writerow(pair)
    return csv_path

class ImageCleaner(BasicImageWidget):
    "Displays images for relabeling or deletion and saves changes in `path` as 'cleaned.csv'."
    def __init__(self, dataset:LabelLists, fns_idxs:Collection[int], path:PathOrStr, batch_size=5, duplicates=False):
        super().__init__(dataset, fns_idxs, batch_size=(2 if duplicates else batch_size),
                         drop_batch_on_nonfile=duplicates, before_next_batch=self.before_next_batch)
        self._duplicates,self._path,self._skipped = duplicates,Path(path),0
        self._csv_dict = {dataset.x.items[i]: dataset.y[i] for i in range(len(dataset))}
        self._deleted_fns:List[Path] = []
        self.next_batch()

    def make_payload(self, idx:int): return {'file_path': self._dataset.x.items[idx]}

    def before_next_batch(self, payloads:Tuple[Mapping]):
        for p in payloads:
            fp = p['file_path']
            if p.get('flagged_for_delete'):
                self.delete_image(fp)
                self._deleted_fns.append(fp)

    def get_widgets(self, batch:Tuple[ImgData]) -> List[widgets.Widget]:
        "Create and format widget set."
        widgets = []
        for i, img in enumerate(batch):
            img_widget = self.make_img_widget(img.jpg_blob)
            if not self._duplicates:
                dropdown = self.make_dropdown_widget(options=self._classes, value=img.label,
                                                     handler=self.relabel, img_idx=i)
            delete_btn = self.make_button_widget('Delete', handler=self.on_delete, img_idx=i)
            widgets.append(self.make_vertical_box(
                (img_widget, delete_btn) if self._duplicates else (img_widget, dropdown, delete_btn)))
        return widgets

    def relabel(self, change):
        "Relabel images by moving from parent dir with old label `class_old` to parent dir with new label `class_new`."
        class_new,class_old = change.new,change.old
        fp = self._get_change_payload(change.owner)['file_path']
        self._csv_dict[fp] = class_new

    def on_delete(self, btn: widgets.Button):
        "Flag this image as delete or keep."
        payload = self._get_change_payload(btn)
        flagged = payload.get('flagged_for_delete', False)
        btn.button_style = "" if flagged else "danger"
        payload['flagged_for_delete'] = not flagged

    def delete_image(self, file_path): del self._csv_dict[file_path]

    def batch_contains_deleted(self, batch:Tuple[ImgData]):
        "Check if current batch contains already deleted images."
        return self._duplicates and any(img.payload['file_path'] in self._deleted_fns for img in batch)

    def write_csv(self):
        # Get first element's file path so we write CSV to same directory as our data
        csv_path = self._path/'cleaned.csv'
        with open(csv_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['name','label'])
            for pair in self._csv_dict.items():
                pair = [os.path.relpath(pair[0], self._path), pair[1]]
                csv_writer.writerow(pair)
        return csv_path

    def render(self, batch:Tuple[ImgData]):
        "Re-render Jupyter cell for batch of images."
        clear_output()
        self.write_csv()
        if not batch:
            if self._skipped>0:
                return display(f'No images to show :). {self._skipped} pairs were '
                    f'skipped since at least one of the images was deleted by the user.')
            return display('No images to show :)')
        if self.batch_contains_deleted(batch):
            self.next_batch()
            self._skipped += 1
        else:
            display(self.make_horizontal_box(self.get_widgets(batch)))
            display(self.make_button_widget('Next Batch', handler=self.next_batch, style="primary"))

class PredictionsCorrector(BasicImageWidget):
    "Displays images for manual inspection and relabelling."
    def __init__(self, dataset:LabelLists, fns_idxs:Collection[int],
                 classes:Sequence[str], labels:Sequence[str], batch_size:int=5):
        super().__init__(dataset, fns_idxs, batch_size, classes=classes, labels=labels)
        self.corrections:Dict[int, str] = {}
        self.next_batch()

    def show_corrections(self, ncols:int, **fig_kw):
        "Shows a grid of images whose predictions have been corrected."
        nrows = ceil(len(self.corrections) / ncols)
        fig, axs = plt.subplots(nrows, ncols, **fig_kw)
        axs, extra_axs = np.split(axs.flatten(), (len(self.corrections),))

        for (idx, new), ax in zip(sorted(self.corrections.items()), axs):
            old = self._get_label(idx)
            self._dataset.x[idx].show(ax=ax, title=f'{idx}: {old} -> {new}')

        for ax in extra_axs:
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    def corrected_labels(self) -> List[str]:
        "Returns labels for the entire test set with corrections applied."
        corrected = list(self._labels)
        for i, l in self.corrections.items(): corrected[i] = l
        return corrected

    def make_payload(self, idx:int): return {'idx': idx}

    def render(self, batch:Tuple[ImgData]):
        clear_output()
        if not batch:
            return display('No images to show :)')
        else:
            display(self.make_horizontal_box(self.get_widgets(batch)))
            display(self.make_button_widget('Next Batch', handler=self.next_batch, style='primary'))

    def get_widgets(self, batch:Tuple[ImgData]):
        widgets = []
        for i, img in enumerate(batch):
            img_widget = self.make_img_widget(img.jpg_blob)
            dropdown = self.make_dropdown_widget(options=self._classes, value=img.label,
                                                 handler=self.relabel, img_idx=i)
            widgets.append(self.make_vertical_box((img_widget, dropdown)))
        return widgets

    def relabel(self, change):
        self.corrections[self._get_change_payload(change.owner)['idx']] = change.new
