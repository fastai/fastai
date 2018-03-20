API Reference
====================

:mod:`fastai.dataset`: Dataset

Dataset
-------
.. currentmodule:: fastai

.. autosummary::

   dataset.BaseDataset
   dataset.FilesDataset
   dataset.FilesArrayDataset
   dataset.FilesIndexArrayDataset
   dataset.FilesNhotArrayDataset
   dataset.FilesIndexArrayRegressionDataset
   dataset.ArraysDataset
   dataset.ArraysIndexDataset
   dataset.ArraysNhotDataset
   dataset.ModelData
   dataset.ImageData
   dataset.ImageClassifierData

Core
----
.. currentmodule:: fastai

.. autosummary::

   core.BasicModel
   core.SingleModel
   core.SimpleNet

AdaptiveSoftmax
---------------
.. currentmodule:: fastai

.. autosummary::

   adaptive_softmax.AdaptiveSoftmax
   adaptive_softmax.AdaptiveLoss


ColumnData
----------
.. currentmodule:: fastai

.. autosummary::

   column_data.PassthruDataset
   column_data.ColumnarDataset
   column_data.ColumnarModelData
   column_data.MixedInputModel
   column_data.StructuredLearner
   column_data.StructuredModel
   column_data.CollabFilterDataset
   column_data.EmbeddingDotBias
   column_data.CollabFilterLearner
   column_data.CollabFilterModel
   
ConvLearner
-----------
.. currentmodule:: fastai

.. autosummary::

   conv_learner.ConvLearner
   conv_learner.ConvnetBuilder

DataLoader
----------
.. currentmodule:: fastai

.. autosummary::

   dataloader.DataLoader

IO
----------
.. currentmodule:: fastai

.. autosummary::

   io.TqdmUpTo

Layer Optimizer
---------------
.. currentmodule:: fastai

.. autosummary::

   layer_optimizer.LayerOptimizer

Layers
---------------
.. currentmodule:: fastai

.. autosummary::

   layers.AdaptiveConcatPool2d
   layers.Lambda
   layers.Flatten

Learner
---------------
.. currentmodule:: fastai

.. autosummary::

   learner.Learner

LM RNN
---------------
.. currentmodule:: fastai

.. autosummary::

   lm_rnn.RNN_Encoder
   lm_rnn.MultiBatchRNN
   lm_rnn.LinearDecoder
   lm_rnn.LinearBlock
   lm_rnn.PoolingLinearClassifier
   lm_rnn.SequentialRNN

Model
---------------
.. currentmodule:: fastai

.. autosummary::

   model.Stepper

NLP
---------------
.. currentmodule:: fastai

.. autosummary::

   nlp.DotProdNB
   nlp.SimpleNB
   nlp.BOW_Learner
   nlp.BOW_Dataset
   nlp.BOW_Learner
   nlp.TextClassifierData
   nlp.LanguageModelLoader
   nlp.RNN_Learner
   nlp.ConcatTextDataset
   nlp.ConcatTextDatasetFromDataFrames
   nlp.LanguageModelData
   nlp.TextDataLoader
   nlp.TextModel
   nlp.TextData

Plots
-----
.. currentmodule:: fastai

.. autosummary::

   plots.ImageModelResults

RNN_Reg
-------
.. currentmodule:: fastai

.. autosummary::

   rnn_reg.LockedDropout
   rnn_reg.WeightDrop
   rnn_reg.EmbeddingDropout

SGDR
-------
.. currentmodule:: fastai

.. autosummary::

   sgdr.LogginCallback
   sgdr.Callback
   sgdr.LossRecorder
   sgdr.LR_Updater
   sgdr.LR_Finder
   sgdr.CosAnneal
   sgdr.CircularLR
   sgdr.SavesBestModel
   sgdr.WeightDecaySchedule

Text
-------
.. currentmodule:: fastai

.. autosummary::

   text.Tokenizer
   text.TextDataset
   text.SortSampler
   text.SortishSampler
   text.LanguageModelLoader
   text.LanguageModel
   text.LanguageModelData
   text.RNN_Learner
   text.TextModel

Text
-------
.. currentmodule:: fastai

.. autosummary::

   transforms.Denormalize
   transforms.Normalize
   transforms.RandomRotateZoom
   transforms.TfmType
   transforms.Transform
   transforms.CoordTransform
   transforms.AddPadding
   transforms.CenterCrop
   transforms.RandomCrop
   transforms.NoCrop
   transforms.Scale
   transforms.RandomScale
   transforms.RandomRotate
   transforms.RandomDihedral
   transforms.RandomFlip
   transforms.RandomLighting
   transforms.RandomBlur
   transforms.CropType
   transforms.Transforms

Utils
-------
.. currentmodule:: fastai

.. autosummary::

   utils.MixIterator

