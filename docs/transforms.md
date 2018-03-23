## [\[source\]](../fastai/transforms.js)

# Classes

### class transforms 

## Functions

* <a href="image_gen">image_gen</a>
* noop

#### <a href="#image_gen">def image_gen</a>

Generate a standard set of transformations

  * code
    ```python
    def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
                tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
        if tfm_y is None: tfm_y=TfmType.NO
        if tfms is None: tfms=[]
        elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
        if sz_y is None: sz_y = sz
        scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
                 else Scale(sz, tfm_y, sz_y=sz_y)]
        if pad: scale.append(AddPadding(pad, mode=pad_mode))

        return Transforms(sz, scale + tfms, normalizer, denorm, crop_type, tfm_y=tfm_y, sz_y=sz_y)
     ```
  * arguments
    * normalizer : 
       image normalizing funciton


