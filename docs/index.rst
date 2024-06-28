.. Python-based Reliability in MRI (PyReliMRI) documentation master file, created by
   sphinx-quickstart on Wed Mar 22 14:27:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyReliMRI's documentation!
=======================================================================
.. figure:: img_png/pyrelimri_logo.png
   :align: center
   :figclass: align-center
   :scale: 40%

Python-based Reliability in MRI (PyReliMRI) is an open-source Python tool to calculate multiple group- and individual-level reliability metrics. This package is designed for researchers using MRI data to easily report reliability estimates in their manuscripts, particularly for multi-run and/or multi-session data.

Several packages exist to address different aspects covered in this package. Specifically, tools are available for calculating similarity coefficients, intraclass correlations (e.g., 3dICC in AFNI), or both (e.g., in Python-based `nipype <https://nipype.readthedocs.io/en/latest/>`_ or Matlab's `fmreli <https://github.com/nkroemer/reliability>`_). Alternatively, Ting Xu offers a `Shiny App <https://tingsterx.shinyapps.io/ReliabilityExplorer/>`_ for calculating univariate and multivariate ICCs from .csv data. However, some flexibility may be limited or certain features unavailable when working with preprocessed MRI data. For example, `ICC_rep_anova` is restricted to ICC(3,1), and `fmreli` requires a Matlab license and does not support tetrachoric correlation, pairwise comparisons across images, or atlas-based reliability estimates.

Our goal is to integrate various functions (see Figure 1) into a single package that can be easily downloaded and imported into Python for universal use.

.. figure:: img_png/pyrelimri_fig.png
   :align: center
   :alt: Figure 1: Available Features within PyReliMRI.
   :figclass: align-center

   Figure 1. Functions within the PyReliMRI Library


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
   timeseries_extract.rst

Citing PyReliMRI
-------------------

  Demidenko, M., Mumford, J., & Poldrack, R. (2024). PyReliMRI: An Open-source Python tool for Estimates of Reliability in MRI Data (2.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.12522260

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
