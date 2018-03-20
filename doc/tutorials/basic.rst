Quick Start
===========

Install fast.ai and pytorch
---------------------------

.. code-block:: python

   # Google Colaboratory
   from os import path
   from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
   platform - '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

   accelerator - 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

.. code-block:: python

   #!pip install -q http://download.pytorch.org/whl/{accelerator}/torch- 
   0.3.0.post4-{platform}-linux_x86_64.whl torchvision
   !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m- 
   linux_x86_64.whl 
   !pip install torchvision

.. code-block:: python

   !pip install fastai

Imports
-------

.. code-block:: python

   import torch 
   torch.cuda.is_available()

    True

.. code-block:: python

   from fastai.imports import *
   from fastai.transforms import *
   from fastai.conv_learner import *
   from fastai.model import *
   from fastai.dataset import *
   from fastai.sgdr import *
   from fastai.plots import *
   from fastai.metrics import *

Loading a dataset
-----------------
Download data
-------------
.. code-block:: python

   !curl -O http://files.fast.ai/data/dogscats.zip

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  817M  100  817M    0     0  31.4M      0  0:00:26  0:00:26 --:--:-- 30.6M


Unzip
-----

.. code-block:: python

   !unzip dogscats.zip

    Archive:  dogscats.zip
       creating: dogscats/
       creating: dogscats/sample/
       creating: dogscats/sample/train/
       creating: dogscats/sample/train/cats/
      inflating: dogscats/sample/train/cats/cat.2921.jpg  
      inflating: dogscats/sample/train/cats/cat.394.jpg  
      inflating: dogscats/sample/train/cats/cat.4865.jpg  ats/train/dogs/dog.3880.jpg  
      inflating: dogscats/train/dogs/dog.3501.jpg  
      inflating: dogscats/train/dogs/dog.7816.jpg  
      inflating: dogscats/train/dogs/dog.19.jpg  
      inflating: dogscats/train/dogs/dog.6934.jpg  
      inflating: dogscats/train/dogs/dog.2449.jpg  
      inflating: dogscats/train/dogs/dog.7655.jpg  
      inflating: dogscats/train/dogs/dog.11037.jpg
      [...]


Training
--------

.. code-block:: python

   !ls

    datalab  dogscats  dogscats.zip

.. code-block:: python

   PATH - "dogscats/"
   sz-224

.. code-block:: python

   arch-resnet34
   data - ImageClassifierData.from_paths(PATH, tfms-tfms_from_model(arch, sz))
   learn - ConvLearner.pretrained(arch, data, precompute-True)
   learn.fit(0.01, 2)

    [0.      0.05905 0.02629 0.99023]
    [1.      0.04276 0.02546 0.99072]
    


Predicting
----------

.. code-block:: python

   # this gives prediction for validation set. Predictions are in log scale
   log_preds - learn.predict()

   # 2000 rows, each row has an array with 2 positions
   #     first array position - probability for cat
   #     second array position - probability for a dog

   log_preds.shape

    (2000, 2)


.. code-block:: python

   # This is the label for a val data
   target_values - data.val_y

.. code-block:: python

   # from array to labels (get the max value for each row, returns 0 if it's the first position [cat] or 1 if it's the second [dog])
   pred_labels - np.argmax(log_preds, axis-1)  

   pred_labels.shape


    (2000,)


.. code-block:: python

   def accuracy_np(preds, targs):
       preds - np.argmax(preds, 1)

       return (preds--targs).mean()
  
   accuracy_np(log_preds, target_values)


    0.9905


.. code-block:: python

   from sklearn.metrics import confusion_matrix
   cm - confusion_matrix(target_values, pred_labels)

.. code-block:: python

   plot_confusion_matrix(cm, data.classes)

    [[992   8]
     [ 11 989]]



