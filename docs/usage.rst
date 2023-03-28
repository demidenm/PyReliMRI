pyrelimri
=========

The `pyrelimri` package contains multiple modules for calculating image reliability measures.

brain_icc
---------

`brain_icc` module contains the following function:

* `voxelwise_icc`: As show in Figure 1, calculates the intraclass correlation (e.g., ICC(1), ICC(2,1), or ICC(3,1)) for 3D volumes across 1+ sessions, returning five 3D volumes reflecting the ICC estimate, the 95% lowerbound for ICC estimate, 95% upperbound for ICC estimate, mean squared error between subjects, mean squared error within subjects.

Inputs: `pyrelimri.icc.voxelwise_icc(paths_sess1, paths_sess2, mask, paths_sess3 = None, icc_type = 'icc_3')`

  * REQUIRED: Paths to 3D Nifti session 1: string, paths to 3D Nifti session 2: string, brain mask: string
  * OPTIONAL: Paths to 3D Nifti session 3 and ICC type (icc_type: string; default = ‘icc_3’, options include: ‘icc_3’,’icc_2’,’icc_1’)

.. figure:: img_png/intraclasscorr_example.png
   :align: center
   :alt: Example ABCD
   :figclass: align-center

   Figure 1. Voxelwise intraclass correlation


icc
---

`icc` module contains the following functions:
* `sumsq_total(df_long, values)`: Calculates to total sum of squared error between subjects & sessions.
* `sumsq_within(df_long, sessions, values, n_subjects)`: Calculates the sum of squared error within subjects across sessions
* `sumsq_btwn(df_long, subj, values, n_sessions)`: Calculates the sum of squared error between subjects
* `icc_confint(msbs, msws, mserr, msc, n_subjs, n_sess, icc_2 = None, alpha = 0.05, icc_type = 'icc_3')`: Calculates the 95% confidence interval (default) using f-statistic
* `sumsq_icc(df_long, sub_var, sess_var, value_var)`: Calculates the total, within, between error and returns ICC estimate (e.g., ICC(1), ICC(2,1), or ICC(3,1)) with associated 95% lowerbound and 95% upperbound for ICC, mean between subject variance and mean within-subject variance.

Inputs:
  * REQUIRED: Panda long dataframe with a subject variable (sub_var: string), session variable (sess_var: string), the scores (value_var: string) and the icc type (icc_type: string; default = default = 'icc_3', options include: 'icc_3', 'icc_2', 'icc_1')



similarity
----------

`similarity` module contains the following functions:

* `image_similarity(imgfile1, imgfile2, mask = None, thresh = None, similarity_type = 'dice')`: Calculates the similarity between two images. For example, in Figure 1a a measure 1 nifti img1 (thresholded p < 001, blue) and img2 (thresholded p <001, green). The overlapping thresholded voxels are in red. By requesting the Jaccard similarity coefficient (Fig1b), you will get a index of similarity between these two nifti images. Alternatively, you may ask what is the similarity using a binary correlation. Using tetrachoric correlation (Fig1c) we can get the similarity between voxels that are above the p < .001 threshold (==1) and those below (==0) between the two images.

* `permute_images(nii_filelist, mask = None, thresh = None, similarity_type = 'dice')`: Calculates the similarity between two images. Permute across 2+ images to calculate similarity coefficient between all possible image pairs.

Inputs:
  * REQUIRED: Path to 3D Nifti files: string, Path to 3D Nifti imgfile2: string, or 3D Nifti file list (nii_filelist: string)
  * OPTIONAL: Path to a mask: string, threshold (thresh: float/integer) on the images, Type (similarity_type: string) of image similarity coefficient (default = ‘dice’, options include: ‘dice’, ‘jaccard’, ‘tetrachoric’)

.. figure:: img_png/similarity_example.png
   :align: center
   :alt: Example ABCD
   :figclass: align-center

   Figure 2. Similarity Between Images

tetrachoric_correlation
-----------------------

`tetrachoric_correlation` module contains the following function:

* `tetrachoric_corr(vec1, vec2)`: Calculates the tetrachoric correlation between two binary vectors.

Inputs:
  * REQUIRED: Binary vector1 NDarray, Binary vector2 NDarray



