Python-based Reliability in MRI (PyReliMRI)
============================================

.. image:: https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml/badge.svg
   :target: https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml

PyReliMRI is a Python package designed to address the increasing interest for reliability assessment in MRI research, \
particularly in `task fMRI <https://www.doi.org/10.1177/0956797620916786>`_ and `resting state fMRI <https://www.doi.org/10.1016/j.neuroimage.2019.116157>`_. \
Researchers use various methods to calculate reliability, but there is a lack of open-source tools that integrate \
multiple metrics for both individual and group analyses.

Purpose of PyReliMRI
---------------------

PyReliMRI (pronounced: Pi-Rely-MRI) aims to fill the gap by providing an open-source Python package for estimating \
multiple reliability metrics on fMRI (or MRI) data in standard space. It supports analysis at both the group and \
individual levels, facilitating comprehensive reporting in multi-run and/or multi-session MRI studies. \
Even with single-session and single-run data, PyReliMRI remains useful. For example:

- Assessing reliability or similarity metrics on individual files by splitting the run and modeling them separately.
- Using group-level maps (e.g., from neurovault or across studies) to compute various similarity metrics.

Modules Overview
-----------------

PyReliMRI comprises several modules tailored to different use cases:

- **`icc`**: Computes various components used in ICC calculations, including ICC(1), ICC(2,1), or ICC(3,1), confidence intervals, between-subject variance, and within-subject variance.
- **`brain_icc`**: Calculates voxelwise and ROI-based ICCs across multiple sessions, integrating with `Nilearn datasets <https://nilearn.github.io/dev/modules/datasets.html>`_ for atlas options.
- **`conn_icc`**: Estimates ICC for precomputed correlation matrices, useful for connectivity studies.
- **`similarity`**: Computes similarity coefficients (Dice, Jaccard, tetrachoric, Spearman) between 3D Nifti images, including pairwise comparisons across multiple images.
- **`tetrachoric_correlation`**: Calculates tetrachoric correlation between binary vectors.
- **`masked_timeseries`**: Extracts and processes timeseries data from BOLD image paths, facilitating ROI-based analysis and event-locked responses.

Each module is designed to answer specific questions about data reliability, supporting a range of MRI analyses in standard spaces like MNI or Talairach.

Citation
---------

If you use PyReliMRI in your research, please cite it using the following Zenodo DOI:

    Demidenko, M., Mumford, J., & Poldrack, R. (2024). PyReliMRI: An Open-source Python tool for Estimates of Reliability in MRI Data (2.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.12522260
