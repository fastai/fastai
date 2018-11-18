import PIL.Image
from IPython.core.display import display, HTML

from ..basic_data import DatasetType
from ..torch_core import *
from ..vision.image import image2np
from ..vision.data import ImageDataBunch

__all__ = ['FacetsDive']


class FacetsDive:

    def __init__(self, data:ImageDataBunch, ds_type:DatasetType=DatasetType.Valid,
                 preds:Union[None, Tuple, List]=None, filter_fn:Callable=None, thumb_size:Tuple[int, int]=(128,128),
                 bg_color_rgb=(0,0,0), facets_path='facets_tmp'):
        """Create a FacetsDive object that can be used to visualize the images.
        For more information about Facets Dive Visualization see https://pair-code.github.io/facets/"""

        # XXX: Elsewhere in the code, the convention is thumbsize. But chose thumb_size to keep it consistent with
        # all other arguments to this function. I don't have a strong preference either way.

        self.data = data
        self.dl = data.dl(ds_type)
        self.classes = data.classes

        if preds is not None:
            # Sanitize preds prevent decode errors when converting to JSON
            self.preds = preds = [o.tolist() if isinstance(o, Tensor) else o for o in preds]
            self.probs = [max(o) for o in preds[0]]
            self.pred_class_idxs = [np.argmax(o) for o in preds[0]]
            self.pred_class_names = [self.classes[o] for o in self.pred_class_idxs]
            self.true_class_idxs = preds[1]
            self.true_class_names = [self.classes[int(o)] for o in self.true_class_idxs]
            self.losses = None if len(preds) == 2 else preds[2]

        self.facets_path, self.thumb_size, self.bg_color_rgb = facets_path, thumb_size, bg_color_rgb
        self.filter_fn = filter_fn
        self.metadata_json = self.metadata = self.path_sprite = self.path_json = None
        self._facets_generated = False

    def _generate_facets(self):
        if not os.path.exists(os.path.normpath(self.facets_path)):
            print(f'{self.facets_path} does not exist. Creating it.')
            #XXX: Warn user to add this to gitignore
            os.mkdir(os.path.normpath(self.facets_path))

        tsize_h, tsize_v = self.thumb_size
        thumbs = {}
        images_metadata = []
        num_samples = 0

        pb = progress_bar(itertools.chain.from_iterable(iter(self.dl.batch_sampler)), total=len(self.dl.dataset))
        pb.comment = 'Generating Thumbnails'
        for sample_idx in pb:
            image, category = self.dl.dataset[sample_idx]
            class_name, class_idx = category.obj, category.data
            sample_name = f'{sample_idx}_{class_name}'
            pred_class_name =  loss = is_pred_accurate = prob = pred_class_idx = None
            if self.preds is not None:
                prob = self.probs[sample_idx]
                pred_class_name = self.pred_class_names[sample_idx]
                pred_class_idx = self.pred_class_idxs[sample_idx]
                true_class_name = self.true_class_names[sample_idx]
                true_class_idx = self.true_class_idxs[sample_idx]
                assert (class_idx == true_class_idx) and (class_name == true_class_name)
                is_pred_accurate = (class_name == pred_class_name)

            if self.losses is not None:
                loss = self.losses[sample_idx]

            if self.filter_fn is not None:
                if not self.filter_fn(
                    sample_idx=sample_idx, sample_name=sample_name, class_idx=class_idx, class_name=class_name,
                    pred_class_name=pred_class_name, pred_class_idx=pred_class_idx, prob=prob,
                        loss=loss, is_pred_accurate=is_pred_accurate):
                    continue

            sample_metadata = {
                'sample_idx': sample_idx, 'sample_name': sample_name, 'class_idx': class_idx, 'class_name': class_name,
                'aspect_ratio': 0, 'width': 0, 'height': 0, 'thumb_width': 0, 'thumb_height': 0,
                'is_vertical': False, 'res': None, 'pred_class_name': pred_class_name, 'pred_class_idx': pred_class_idx,
                'loss': loss, 'prob': prob, 'is_pred_accurate': is_pred_accurate}

            x = image2np(image.data * 255).astype(np.uint8)

            pil_img = PIL.Image.fromarray(x)
            w, h = pil_img.size

            pil_img.thumbnail((tsize_h, tsize_v))  # Rescales it in place
            thumb_w, thumb_h = pil_img.size

            # Save a reference to the rescaled thumbnail in the thumbs dict.
            # We'll need this later when we generate sprite.
            thumbs[sample_idx] = pil_img

            sample_metadata['width'], sample_metadata['height'], sample_metadata['aspect_ratio'] = w, h, w * 1.0 / h
            sample_metadata['thumb_width'], sample_metadata['thumb_height'] = thumb_w, thumb_h
            sample_metadata['is_vertical'] = (h > w)
            images_metadata.append(sample_metadata)
            num_samples += 1

        max_width = max(x['width'] for x in images_metadata)
        max_height = max(x['height'] for x in images_metadata)

        for o in images_metadata:
            fmt_str = f'%0{len(str(max_width))}dx%0{len(str(max_height))}d'
            o['res'] = fmt_str % (o['width'], o['height'])

        # Figure out the max sizes of thumbnails to generate a sprite of appropriate size
        thumb_max_width = max(x['thumb_width'] for x in images_metadata)
        thumb_max_height = max(x['thumb_height'] for x in images_metadata)
        num_cols = int(math.floor(math.sqrt(num_samples)))
        num_rows = int(math.ceil(num_samples * 1.0 / num_cols))
        sprite_size = (thumb_max_width * num_cols, thumb_max_height * num_rows)
        # Generate a blank sprite
        sprite = PIL.Image.new(mode='RGB', size=sprite_size, color=self.bg_color_rgb)
        # Paste thumbnails into sprite
        pbar = progress_bar(enumerate(images_metadata), total=num_samples)
        pbar.comment = 'Creating Sprite Image'
        for i, d in pbar:
            top = (i // num_cols) * thumb_max_height
            left = (i % num_cols) * thumb_max_width
            thumb = thumbs[d['sample_idx']]
            sprite.paste(thumb, (left, top))
            thumb.close()
        # Hash the image to get the fname
        fname = hashlib.md5(sprite.tobytes()).hexdigest()
        # Save Sprite file
        # NOTE: The HTML server cannot access paths that are outside of the directory tree from which
        # jupyter notebook server is running. For example, if jupyter notebook is running at
        # /home/johndoe/notebooks/example
        # the file at /tmp/images/image1.jpg cannot be served.
        sprite_fpath = os.path.join(self.facets_path, fname + '.jpg')
        sprite.save(sprite_fpath, 'JPEG')
        sprite.close()

        # Convert DataFrame to because json.dump raises Numpy datatype decode errors
        # Using Pandas DF is an easy way to ensure Numpy errors are handled transparently
        self.metadata = images_metadata
        df = pd.DataFrame(self.metadata)
        self.metadata_json = df.to_json(orient='records')
        with open(os.path.join(self.facets_path, fname + '.json'), 'w') as f:
            f.write(self.metadata_json)
        self.path_json = Path(os.path.join(self.facets_path, fname+'.json'))
        self.path_sprite = Path(os.path.join(self.facets_path, fname+'.jpg'))

        self._facets_generated = True

    def show(self):

        if not self._facets_generated:
            self._generate_facets()

        facets_dive_html = f"""
            <link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html">
            <facets-dive
                id="elem"
                height="600"
                sprite-image-width="128"
                sprite-image-height="128"
                atlas-url="{self.path_sprite}" >
            </facets-dive>
            <script>
                var data = {self.metadata_json};
                document.querySelector("#elem").data = data;
            </script>
        """
        display(HTML(facets_dive_html))
