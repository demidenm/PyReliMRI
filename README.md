[![Python package](https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/demidenm/PyReliMRI/actions/workflows/python-package-conda.yml)

# Python-based Reliability in MRI (PyReliMRI)

PyReliMRI is described and applied in the [TBD Preprint](https://www.doi.org)

## Authors

[Michael I. Demidenko](https://orcid.org/0000-0001-9270-0124) & [Russell A. Poldrack](https://orcid.org/0000-0001-6755-0259)

## Intro of Problem

Reliability questions for [task fMRI](https://doi.org/10.1177/0956797620916786) and [resting state fMRI](www.doi.org/10.1016/j.neuroimage.2019.116157) are increasing. As described in [2010](www.doi.org/10.1111/j.1749-6632.2010.05446.x), there are various ways that researchers calculate reliability. Few open-source packages exist to calculate multiple individual and group reliability metrics using one tool.

## Purpose of Script

The purpose of this package is to provide an open-source python package that will provide multiple reliability metrics, at the group and individual level, that researchers may use to report in their manuscripts in cases of multi-run and/or multi-session data.
 - At the group level, [similarity.py](/imgreliability/similarity.py) calculates a similarity calculations between two fMRI images using Dice or Jaccard similarity coefficients or tetrachoric correlation. In addition to calculate the similarity between two images, a function is provided to permute across a list of 3D images and return a list of coefficients.
 - At the individual level, the [brain_icc.py](/imgreliability/brain_icc.py) calculates ICC(1), ICC(2,1) or ICC(3,1). For description of different ICCs and their calculations, see discussion in [Liljequist et al., 2019](www.doi.org/10.1371/journal.pone.0219854).


| **Script Name** | **Functions** | **Inputs**                                                                                                                                                                                                                                 | **Purpose**                                                                                                                                                                                                                                                                  |
| :-------------- | :----------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [brain_icc.py](/imgreliability/brain_icc.py) | voxelwise_icc | REQUIRED:<br>paths to 3D session 1,<br>paths to 3D session 3,<br>brain mask<br>OPTIONAL:<br>paths to session 3 and icc type (e.g., ICC(1), ICC(2,1), ICC(3,1))                                                                             | Calculate the intraclass correlation for 3D volumes across 1+ sessions, returning five 3D volumes reflecting the ICC estimate, the 95% lowerbound for ICC estimate, 95% upperbound for ICC estimate, mean squared error between subjects, mean squared error within subjects |
| [icc.py](/imgreliability/icc.py) | sumsq_total,<br>sumsq,<br>sumsq_btwn,<br>icc_confint,<br>sumsq_icc | REQUIRED:<br>Panda long dataframe with a subject variable,<br>session variable,<br>the scores and the icc type                                                                                                                             | Calculates sum of squares total, error, within and between to return an ICC estimate, 95% lowerbound and 95% upperbound for ICC, mean between subject variance and mean within-subject variance                                                                              |
| [similarity.py](/imgreliability/similarity.py) | image_similarity,<br>permute_images | REQUIRED:<br>Path to 3D Nifti IMG File 1,<br>Path to 3D Nifti IMG File 2<br>OPTIONAL:<br>Path to a mask,<br>Threshold on the images,<br>Type of image similarity coefficient (default = Dice, options include: Dice, Jaccard, Tetrachoric) | Calculate the similarity between two images. Permute multiple images to calculate similarity coefficient between all possible image pairs.                                                                                                                                   |
| [tetrachoric_correlation.py](/imgreliability/tetrachoric_correlation.py) | tetrachoric_corr | REQUIRED:<br>Binary IMG1 NDarray,<br>Binary IMG2 NDarray                                                                                                                                                                                   | Calculate the tetrachoric correlation between two binary images.                                                                                                                                                                                                             |
