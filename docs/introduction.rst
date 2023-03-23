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

The purpose of this package is to provide an open-source python package that will provide multiple reliability metrics, at the group and individual level, that researchers may use to report in their manuscripts in cases of multi-run and/or multi-session data.
At the group level, ``similarity.py`` calculates a similarity calculations between two fMRI images using Dice or Jaccard similarity coefficients or tetrachoric correlation. In addition to calculate the similarity between two images, a function is provided to permute across a list of 3D images and return a list of coefficients.

At the individual level, the ``brain_icc.py`` calculates ICC(1), ICC(2,1) or ICC(3,1). For description of different ICCs and their calculations, see discussion in `Liljequist et al., 2019 <https://www.doi.org/10.1371/journal.pone.0219854>`_.

.. list-table::
   :header-rows: 1
   :widths: 15, 20, 50, 80
   :class: wrap

   * - Name
     - Functions
     - Inputs
     - Purpose

   * - brain_icc.py
     - voxelwise_icc
     - **REQUIRED:** Paths to 3D Nifti session 1,paths to 3D Nifti session 2, brain mask  **OPTIONAL:** paths to 3D Nifti session 3 and ICC type (icc_type; default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')
     - Calculate the intraclass correlation (e.g., ICC(1), ICC(2,1), or ICC(3,1) for 3D volumes across 1+ sessions, returning five 3D volumes reflecting the ICC estimate, the 95% lowerbound for ICC estimate, 95% upperbound for ICC estimate, mean squared error between subjects, mean squared error within subjects)

   * - icc.py
     - sumsq_total, sumsq, sumsq_btwn, icc_confint, sumsq_icc
     - **REQUIRED:** Panda long dataframe with a subject variable (sub_var), session variable (sess_var), the scores (value_var) and the icc type (icc_type; default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')
     - Calculates sum of squares total, error, within and between to return an ICC estimate (e.g., ICC(1), ICC(2,1), or ICC(3,1), 95% lowerbound and 95% upperbound for ICC, mean between subject variance and mean within-subject variance)

   * - similarity.py
     - image_similarity,permute_images
     - **REQUIRED:** Path to 3D Nifti imgfile1,Path to 3D Nifti imgfile2 **OPTIONAL:** Path to a mask, threshold (thresh) on the images, Type (similarity_type) of image similarity coefficient (default = 'dice', options include: 'dice', 'jaccard', 'tetrachoric')
     - Calculate the similarity between two images. Permute multiple images to calculate similarity coefficient between all possible image pairs.

   * - tetrachoric_correlation.py
     - tetrachoric_corr
     - **REQUIRED:** Binary vector NDarray,Binary vector NDarray
     - Calculate the tetrachoric correlation between two binary vectors.
