.. Python-based Reliability in MRI (PyReliMRI) documentation master file, created by
   sphinx-quickstart on Wed Mar 22 14:27:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyReliMRI's documentation!
=======================================================================
Python-based Reliability in MRI (PyReliMRI) is an open-source python \
tool to calculate multiple group- and individual-level reliability metrics. This package is released \
for researchers using MRI data to easily report reliability estimates in their manuscripts in cases of multi-run \
and/or multi-session data are acquired.

There are a number of packages available to achieve different aspects contained in this package. \
Specifically, tools exist to calculate either simiarity coefficient, intraclass corelations (e.g., 3dICC in AFNI) \
or both (e.g., in python based `nipype <https://nipype.readthedocs.io/en/latest/>`_ or matlab `fmreli <https://github.com/nkroemer/reliability>`_). \
However, in some cases the flexibility is limited or some features are not available. \
For example, `ICC_rep_anova` is limited to ICC(3,1) and the `fmreli` is not accessible without \
a matlab license and does not deploy the the tetrachoric correlation or permutation across images.

Our attempt here is to integrate different functions within the same package that anyone can use by \
downloading python and importing the package on their machine.


PyReliMRI
====================================

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   introduction.rst

Module Documentation
====================================

.. toctree::
   :maxdepth: 2
   :caption: Usage:

   install.rst
   usage.rst

Usage Examples
====================

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples.rst
   similarity_funcs.rst
   icc_funcs.rst

Citing PyReliMRI
-------------------

  Demidenko, M.I. & Poldrack, R.A. [TBD].

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
