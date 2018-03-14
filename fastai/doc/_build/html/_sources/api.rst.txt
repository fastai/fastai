API reference
====================

Dataset
----
.. currentmodule:: fastai.fastai.dataset

.. autosummary::
   :toctree: generated/

   fastai.fastai.dataset.BaseDataset
   fastai.fastai.dataset.FilesDataset
   fastai.fastai.dataset.FilesArrayDataset
   fastai.fastai.dataset.FilesIndexArrayDataset
   fastai.fastai.dataset.FilesNhotArrayDataset
   fastai.fastai.dataset.FilesIndexArrayRegressionDataset
   fastai.fastai.dataset.ArraysDataset
   fastai.fastai.dataset.ArraysIndexDataset
   fastai.fastai.dataset.ArraysNhotDataset
   fastai.fastai.dataset.ModelData
   fastai.fastai.dataset.ImageData
   fastai.fastai.dastaset.ImageClassifierData

Core
----
.. currentmodule:: fastai.fastai.core

.. autosummary::

   fastai.fastai.core.BasicModel
   fastai.fastai.core.SingleModel
   fastai.fastai.core.SimpleNet

AdaptiveSoftmax
---------------
.. currentmodule:: fastai.fastai.adaptive_softmax

.. autosummary::
   :toctree: 

   fastai.fastai.adaptive_softmax.AdaptiveSoftmax
   fastai.fastai.adaptive_softmax.AdaptiveLoss


ColumnData
----------
.. currentmodule:: fastai.fastai.column_data

.. autosummary::
   :toctree: 

   fastai.fastai.column_data.PassthruDataset
   fastai.fastai.column_data.ColumnarDataset
   fastai.fastai.column_data.ColumnarModelData
   fastai.fastai.column_data.MixedInputModel
   fastai.fastai.column_data.StructuredLearner
   fastai.fastai.column_data.StructuredModel
   fastai.fastai.column_data.CollabFilterDataset
   fastai.fastai.column_data.EmbeddingDotBias
   fastai.fastai.column_data.CollabFilterLearner
   fastai.fastai.column_data.CollabFilterModel
   
ConvLearner
----------
.. currentmodule:: fastai.fastai.conv_learner

.. autosummary::
   :toctree: 

   fastai.fastai.conv_learner.ConvLearner
   fastai.fastai.conv_learner.ConvnetBuilder

DataLoader
----------
.. currentmodule:: fastai.fastai.dataloader

.. autosummary::
   :toctree: 

   fastai.fastai.dataloader.DataLoader

IO
----------
.. currentmodule:: fastai.fastai.io

.. autosummary::
   :toctree: 

   fastai.fastai.io.TqdmUpTo

Layer Optimizer
---------------
.. currentmodule:: fastai.fastai.layer_optimizer

.. autosummary::
   :toctree: 

   fastai.fastai.layer_optimizer.LayerOptimizer

Layers
---------------
.. currentmodule:: fastai.fastai.layers

.. autosummary::
   :toctree: 

   fastai.fastai.layers.AdaptiveConcatPool2d
   fastai.fastai.layers.Lambda
   fastai.fastai.layers.Flatten

Learner
---------------
.. currentmodule:: fastai.fastai.learner

.. autosummary::
   :toctree: 

   fastai.fastai.learner.Learner

LM RNN
---------------
.. currentmodule:: fastai.fastai.lm_rnn

.. autosummary::
   :toctree: 

   fastai.fastai.lm_rnn.RNN_Encoder
   fastai.fastai.lm_rnn.MultiBatchRNN
   fastai.fastai.lm_rnn.LinearDecoder
   fastai.fastai.lm_rnn.LinearBlock
   fastai.fastai.lm_rnn.PoolingLinearClassifier
   fastai.fastai.lm_rnn.SequentialRNN

Model
---------------
.. currentmodule:: fastai.fastai.model

.. autosummary::
   :toctree: 

   fastai.fastai.model.Stepper

NLP
---------------
.. currentmodule:: fastai.fastai.nlp

.. autosummary::
   :toctree: 

   fastai.fastai.nlp.DotProdNB
   fastai.fastai.nlp.SimpleNB
   fastai.fastai.nlp.BOW_Learner
   fastai.fastai.nlp.BOW_Dataset
   fastai.fastai.nlp.BOW_Learner
   fastai.fastai.nlp.TextClassifierData
   fastai.fastai.nlp.LanguageModelLoader
   fastai.fastai.nlp.RNN_Learner
   fastai.fastai.nlp.ConcatTextDataset
   fastai.fastai.nlp.ConcatTextDatasetFormDataFrames
   fastai.fastai.nlp.LanguageModelData
   fastai.fastai.nlp.TextDatLoader
   fastai.fastai.nlp.TextModel
   fastai.fastai.nlp.TextData

Plots
-----
.. currentmodule:: fastai.fastai.plots

.. autodummary::
   :toctree:

   fastai.fastai.plots.ImageModelResults

RNN_Reg
-------
.. currentmodule:: fastai.fastai.rnn_reg

.. autodummary::
   :toctree:

   fastai.fastai.rnn_reg.LockedDropout
   fastai.fastai.rnn_reg.WeightDrop
   fastai.fastai.rnn_reg.EmbeddingDropout

SGDR
-------
.. currentmodule:: fastai.fastai.sgdr

.. autodummary::
   :toctree:

   fastai.fastai.sgrd.LogginCallback
   fastai.fastai.sgrd.Callback
   fastai.fastai.sgrd.LossRecorder
   fastai.fastai.sgrd.LR_Updater
   fastai.fastai.sgrd.LR_Finder
   fastai.fastai.sgrd.CosAnneal
   fastai.fastai.sgrd.CircularLR
   fastai.fastai.sgrd.SavesBestModel
   fastai.fastai.sgrd.WeightDecaySchedule

Text
-------
.. currentmodule:: fastai.fastai.text

.. autodummary::
   :toctree:

   fastai.fastai.text.Tokenizer
   fastai.fastai.text.TextDataset
   fastai.fastai.text.SortSampler
   fastai.fastai.text.SortishSamples
   fastai.fastai.text.LanguageModelLoader
   fastai.fastai.text.LanguageModel
   fastai.fastai.text.LanguageModelData
   fastai.fastai.text.RNN_Learner
   fastai.fastai.text.TextModel

Text
-------
.. currentmodule:: fastai.fastai.transforms

.. autodummary::
   :toctree:

   fastai.fastai.transforms.Denormalize
   fastai.fastai.transforms.Normalize
   fastai.fastai.transforms.RandomRotateZoom
   fastai.fastai.transforms.TfmType
   fastai.fastai.transforms.Transform
   fastai.fastai.transforms.CoordTransform
   fastai.fastai.transforms.AddPadding
   fastai.fastai.transforms.CenterCrop
   fastai.fastai.transforms.RandomCrop
   fastai.fastai.transforms.NoCrop
   fastai.fastai.transforms.Scale
   fastai.fastai.transforms.RandomScale
   fastai.fastai.transforms.RandomRotate
   fastai.fastai.transforms.RandomDihedral
   fastai.fastai.transforms.RandomFlip
   fastai.fastai.transforms.RandomLighting
   fastai.fastai.transforms.RandomBlur
   fastai.fastai.transforms.CropType
   fastai.fastai.transforms.Transforms

Utils
-------
.. currentmodule:: fastai.fastai.utils

.. autodummary::
   :toctree:

   fastai.fastai.utils.MixIterator

