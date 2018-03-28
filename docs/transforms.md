## Classes

* [Normalize](#normalize)
* [Denormalize](#denormalize)

#### <a name="normalize">class Normalize</a>
Normalizes an image
  * code
    ```python
    class Normalize():
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None:
            y = (y-self.m)/self.s
        return x,y
        
    ```
  * arguments
  
#### <a name="denormalize">class Denormalize</a>

Denormalizes an image
  * code
    ```python
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m
    ```
  * arguments


## Functions

* [image_gen](#image_gen)

#### <a name="image_gen">def image_gen</a>

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
    * denorm:  
       image denormalizing function
    * sz:  
      size, sz_y = sz if not specified.  
    * tfms:  
      iterable collection of transformation functions
    * max_zoom:  
      maximum zoom
    * pad:  
      padding on top, left, right and bottom
    * crop_type:  
      crop type
    * tfm_y:  
      y axis specific transformations
    * sz_y:  
      y size, height
    * pad_mode:  
      cv2 padding style: repeat, reflect, etc. 

