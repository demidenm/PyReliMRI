Description
===========

The PyReliMRI package intergrates several modules designed to facilitate reliability estimation on MRI data. \
The code is simplified by leveraging features from `Nilearn <https://nilearn.github.io/stable/index.html>`_. \
These modules can be categorized into two main groups:

Similarity and Tetrachoric Correlation
---------------------------------------

- `similarity.py`: Computes similarity coefficients (Dice, Jaccard, etc.) between 3D Nifti images. Includes functions like `image_similarity` for pairwise comparisons.

- `tetrachoric_correlation.py`: Calculates the tetrachoric correlation between binary vectors, useful for certain types of data analysis.

Intraclass Correlation
-----------------------

- `icc.py`: Computes various components used in ICC calculations, such as ICC(1), ICC(2,1), or ICC(3,1), along with confidence intervals and variance components.

- `brain_icc.py`: Calculates voxelwise and ROI-based ICCs across multiple runs/sessions. Integrates with Nilearn datasets for atlas options, facilitating quick atlas integration.

- `conn_icc.py`: Estimates ICC for precomputed correlation matrices, useful for connectivity analyses.

Stimulus-Locked TR-by-TR Timeseries
-------------------------------------

The `masked_timeseries.py` module provides functionality for extracting and processing stimulus-locked timeseries data from BOLD images. It includes methods for ROI-based analysis and event-locked responses.


Combined, these modules collectively support a wide range of reliability assessments in MRI studies, from basic similarity metrics to advanced ICC calculations and timeseries analysis.
