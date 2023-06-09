Python-based Reliability in MRI (PyReliMRI)
-------------------------------------------

.. image:: https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml/badge.svg
    :target: https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml


PyReliMRI is described and applied in the `TBD Preprint <https://www.doi.org>`_.

Authors
~~~~~~~

- `Michael I. Demidenko <https://orcid.org/0000-0001-9270-0124>`_
- `Russell A. Poldrack <https://orcid.org/0000-0001-6755-0259>`_

Intro of Problem
~~~~~~~~~~~~~~~~~

Reliability questions for `task fMRI <https://https://www.doi.org/10.1177/0956797620916786>`_ and `resting state fMRI <https://www.doi.org/10.1016/j.neuroimage.2019.116157>`_ are increasing. As described in `2010 <https://www.doi.org/10.1111/j.1749-6632.2010.05446.x>`_, there are various ways that researchers calculate reliability. Few open-source packages exist to calculate multiple individual and group reliability metrics using one tool.

Purpose of Script
~~~~~~~~~~~~~~~~~~

The purpose of this PyReliMRI is to provide an open-source python package that will estimate multiple reliability \
metrics on fMRI data -- at the group and individual level -- \
that researchers may use to report in their manuscripts in cases of multi-run and/or multi-session MRI data.

At the group level, ``similarity.py`` calculates the similarity between two 3D Nifti images using Dice or Jaccard \
similarity coefficients, or tetrachoric or spearman ocorrelation. In addition to calculating the similarity between two NifTi images \
a `pairwise_similarity` option is available to calculate pairwise similarity coefficients across a list of \
3D Nifti images and return a list of coefficients with associated image labels.

At the individual level, the ``brain_icc.py`` calculates intraclass correlation. For description of different ICCs and their calculations, \
see discussion in `Liljequist et al., 2019 <https://www.doi.org/10.1371/journal.pone.0219854>`_. In this package, you have the option to \
select ICC(1), ICC(2,1) or ICC(3,1).

.. list-table::
   :header-rows: 1
   :widths: 15, 20, 50, 80
   :class: wrap

   * - Name
     - Functions
     - Inputs
     - Purpose

   * - brain_icc
     - voxelwise_icc
     - **REQUIRED:** list of session 1, session 2, etc, paths to 3D NifTi images, path to 3D NifTi brain mask  **OPTIONAL:** ICC type (icc_type; default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')
     - Calculates and returns dictionary with intraclass correlation (e.g., ICC(1), ICC(2,1), or ICC(3,1) for 3D volumes across 1+ sessions, reflecting the ICC estimate, the 95% lowerbound for ICC estimate, 95% upperbound for ICC estimate, mean squared error between subjects, mean squared error within subjects)
     - Note: Ensure that your mask contains all voxels for subjects. If voxels are NaN or zero for some subjects, NaN mean-based replacement is used and/or zeros are treated as true observed zeros.

   * - icc
     - sumsq_total, sumsq, sumsq_btwn, icc_confint, sumsq_icc
     - **REQUIRED:** Panda long dataframe with a subject variable (sub_var), session variable (sess_var), the scores (value_var) and the icc type (icc_type; default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')
     - Calculates different components used in calculating the ICC estimate (e.g., ICC([1], ICC([2,1], or ICC[3,1]), 95% lowerbound and 95% upperbound for ICC, mean between subject variance and mean within-subject variance
     - Note: If NaN/missing values, unlike ICC in Psych package in R and intraclass_corr in Pingouin in Python which use listwise deletion, use mean replaces of all column values.
     - If an alternate method is preferred, recommend handling missing data & re-balancing prior to use. Significant impact on estimates at high rate of missing and/or low subject N


   * - similarity
     - image_similarity,pairwise_similarity
     - **REQUIRED:** Path to 3D NifTi imgfile1, Path to 3D NifTi imgfile2 **OPTIONAL:** Path to a NifTi mask, threshold level (thresh) on the images, Type (similarity_type) of image similarity coefficient (default = 'dice', options include: 'dice', 'jaccard', 'tetrachoric', 'spearman')
     - Calculates and returns the similarity between two images. Calculates similarity coefficient for 2+ pairwise similarity for all possible image pairs and returns a dataframe.

   * - tetrachoric_correlation.py
     - tetrachoric_corr
     - **REQUIRED:** Binary vector NDarray,Binary vector NDarray
     - Calculate and return tetrachoric correlation between two binary vectors.
