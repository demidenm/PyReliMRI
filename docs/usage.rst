pyrelimri
=========

The `pyrelimri` module contains four `.py` scripts for calculating image reliability measures.

brain_icc.py
------------

`brain_icc.py` contains the following function:

* `voxelwise_icc`: As show in Figure 1, calculates the intraclass correlation (e.g., ICC(1), ICC(2,1), or ICC(3,1)) for 3D volumes across 1+ sessions, returning five 3D volumes reflecting the ICC estimate, the 95% lowerbound for ICC estimate, 95% upperbound for ICC estimate, mean squared error between subjects, mean squared error within subjects.

Inputs:
  * REQUIRED: Paths to 3D Nifti session 1, paths to 3D Nifti session 2, brain mask
  * OPTIONAL: Paths to 3D Nifti session 3 and ICC type (icc_type; default = ‘icc_3’, options include: ‘icc_3’,’icc_2’,’icc_1’)

.. figure:: img_png/intraclasscorr_example.png
   :align: center
   :alt: Example ABCD
   :figclass: align-center

   Figure 1. Voxelwise intraclass correlation


icc.py
------

`icc.py` contains the following function:

* `sumsq_total`: Calculates to total sum of squared error between subjects & sessions
* `sumsq_within`: Calculates the sum of squared error within subjects across sessions
* `sumsq_btwn`: Calculates the sum of squared error between subjects
* `icc_confint`: Calculates the 95% confidence interval (default) using f-statistic
* `sumsq_icc`: Calculates the total, within, between error and returns ICC estimate (e.g., ICC(1), ICC(2,1), or ICC(3,1)) with associated 95% lowerbound and 95% upperbound for ICC, mean between subject variance and mean within-subject variance.

Inputs:
  * REQUIRED: Panda long dataframe with a subject variable (sub_var), session variable (sess_var), the scores (value_var) and the icc type (icc_type; default = default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')



similarity.py
-------------

`similarity.py` contains the following function:

* `image_similarity`: Calculates the similarity between two images. For example, in Figure 1a a measure 1 nifti img1 (thresholded p < 001, blue) and img2 (thresholded p <001, green). The overlapping thresholded voxels are in red. By requesting the Jaccard similarity coefficient (Fig1b), you will get a index of similarity between these two nifti images. Alternatively, you may ask what is the similarity using a binary correlation. Using tetrachoric correlation (Fig1c) we can get the similarity between voxels that are above the p < .001 threshold (==1) and those below (==0) between the two images.

* `permute_images`: Calculates the similarity between two images. Permute across 2+ images to calculate similarity coefficient between all possible image pairs.

Inputs:
  * REQUIRED: Path to 3D Nifti imgfile1, Path to 3D Nifti imgfile2
  * OPTIONAL: Path to a mask, threshold (thresh) on the images, Type (similarity_type) of image similarity coefficient (default = ‘dice’, options include: ‘dice’, ‘jaccard’, ‘tetrachoric’)

.. figure:: img_png/similarity_example.png
   :align: center
   :alt: Example ABCD
   :figclass: align-center

   Figure 2. Similarity Between Images

tetrachoric_correlation.py
--------------------------

`tetrachoric_correlation.py` contains the following function:

* `tetrachoric_corr`: Calculates the tetrachoric correlation between two binary vectors.

Inputs:
  * REQUIRED: Binary vector1 NDarray, Binary vector2 NDarray



