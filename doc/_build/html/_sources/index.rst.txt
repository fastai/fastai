.. fastai documentation master file, created by
   sphinx-quickstart on Tue Feb  6 10:37:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

fast.ai
=======

The fast.ai deep learning library

Why?
----
To build models that are faster, more accurate and more complex with much less code.

How?
----
We created an OO class which encapsulates all of the important data choices (such as preprocessing, augmentation, test, training, and validation sets, multiclass versus singles classification versus regression, etc.) along with the choice of the model artchutecture.

So what?
--------
In doing so we are able to largely automatically figure out the best architecture, preprocessing, and training parameters for that model that model and for that data. Everything that could be automated was automated. 

But we also provide the ability to customise every stage, so we could easily experiment with different approaches.

.. toctree::
   :maxdepth: 1
    
   readme
   tutorials/basic.rst
   tutorials/index
   api 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`



