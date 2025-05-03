.. SOLT documentation master file, created by
   sphinx-quickstart on Fri Sep  7 13:50:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to solt's documentation!
================================

**solt** is a fast data augmentation library, supporting arbitrary amount of images,
segmentation masks, keypoints and data labels.
It has OpenCV in its back-end, thus it works very fast.

Features
--------
* Geometric transformations fusion to speed up a series of arbitrary transformations.
* Logging of all transformations for reproducibility.
* Support of any Computer Vision task.
* Easy extendability, interfacing with other libraries.
* High parametrization.
* Nearly 100% code coverage.

How to install
--------------
The most recent stable version is available in pip:

::

   pip install solt

You can also fetch the most fresh changes from our GitHub repository:

::

   pip install -U git+https://github.com/imedslab/solt.git



.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 3
   :caption: Modules:

   modules

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   Basics_of_solt
   Helen_faces
   DSBowl18_segmentation
   Medical_Data_Augmentation_CXR14.nblink
