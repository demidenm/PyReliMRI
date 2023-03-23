Intraclass Correlation Functions
=======================================

The intraclass correlation (ICC) estimates are a complement to the similarity functions. The variability/similarity \
in the data can be parsed in several ways. One can estimate home similar things are about a threshold (e.g., 1/0) or \
how similarity specific continuous estimates are across subjects. The ICC is used for the latter here.

Two components are described with some examples: The `icc.py` and `brain_icc.py`. The first is the manual estimation \
of the components of the ICC, such the the sum of (1) squared total,  the sum of (2) squared within, (3) sum of squared between \
their associated mean sums of squared (which is 1-3 divided by the degrees of freedom) in `icc.py`. Then, the next step is to \
calculate these values on a voxel-by-voxel basis (or if you wanted to, ROI by ROI) using `brain.py`.


icc.py
------

While `icc.py` is within the package for MRI reliability, it can still be used to calculate different values on dataframes. \
Below will describe the different components and use `seaborns.anagrams <https://github.com/mwaskom/seaborn-data/blob/master/anagrams.csv>`_ \
as the example for each of these.

The first 5-rows of the anagrams data are:

+--------+---------+-----+-----+-----+
| subidr |  attnr  | num1| num2| num3|
+========+=========+=====+=====+=====+
|    1   | divided |  2  |  4  |  7  |
+--------+---------+-----+-----+-----+
|    2   | divided |  3  |  4  |  5  |
+--------+---------+-----+-----+-----+
|    3   | divided |  3  |  5  |  6  |
+--------+---------+-----+-----+-----+
|    4   | divided |  5  |  7  |  5  |
+--------+---------+-----+-----+-----+
|    5   | divided |  4  |  5  |  8  |
+--------+---------+-----+-----+-----+

We can load the example data, filter to only use the `divided` values and convert the data into a long data format:

.. code-block:: python

    import seaborn as sns
    data = sns.load_dataset('anagrams') # load
    a_wd = data[data['attnr'] == 'divided'] # filter
    # convert to wide, so the subject/id variables are still `subidr` and values were stacking from `num1`,`num2`,num3`
    # the values will be stored in the column `vals` and the session labels (from num1-num3) into `sess`
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
               id_vars="subidr",
               value_vars=["num1", "num2", "num3"],
               var_name="sess",
               value_name="vals"))

**sumsq_total**

The sum of squared total is the estimate of the total variance across all subjects and measurement occasions. Expressed \
the formula used is:

.. math::

    \text{sumsq\_total(df\_long, values)} = \sum_{i=1}^{n}(x_i - \bar{x})^2

where:
    * df_long = pandas DataFrame (df) in long format \
    * values = is a variable string for the values containing the scores in df \
    * x_i = is each value in the column specified by values column in df \
    * x_bar = is the global mean specified by 'values' column in df

Using the anagrams `long_df` we can calculate the sum of square total using:
.. code-block:: python

    from imgreliability import icc
    icc.sumsq_total(df_long=long_df, values="vals")

We will get the result of 71.8 sum of squared `total`.

**sumsq_within**


.. math::

    \text{sumsq_within}(df_{long}, sessions, values, n_{subjects}) = n_{subjects} \sum_{i=1}^m (\overline{x}_i - \overline{x})^2

where:
    * df_long = pandas DataFrame in long format \
    * sessions = is a session (repeated measurement) variable, string, in df \
    * values = is a variable, string, for the values containing the scores in df \
    * n_subjects = the number of subjects in df \
    * x_i_bar = is the mean of the `values` column for session `i` in df \
    * x_bar = is the global mean specified by 'values' column in df
    * m = is the number of sessions


We can calculate the sum of squares within using the below:

.. code-block:: python

    # if you havent imported the package already
    from imgreliability import icc
    icc.sumsq_within(df_long=a_ld,sessions="sess", values="vals", n_subjects=10)

We will get the result of 29.2 sum of squares `between` subject factor.

**sumsq_btwn**

.. math::

    \text{sumsq_btwn}(df_{long}, subj, values, n_{sessions}) = n_{sessions} \sum_{i=1}^s (\overline{x}_i - \overline{x})^2

where:
    * df_long = pandas DataFrame in long format \
    * subj = is the subject variable, string, in df \
    * values = is a variable, string, for the values containing the scores in df \
    * n_sessions = the number of sessions in df \
    * x_i_bar = is the mean of the `values` column for subject `i` in df \
    * x_bar = is the global mean specified by 'values' column in df
    * s = is the number of subjects

.. code-block:: python

    # if you havent imported the package already
    from imgreliability import icc
    icc.sumsq_btwn(df_long=a_ld,subj="subidr", values="vals", n_sessions=3) # 3 = num1-num3

We will get the result of 20.0 sum of squares `between` subject factor.

Note: If you recall that ICC is the decomposition of `total` variance, you'll notice that 29.2 + 20.0 \
do not sum to the total variance, 71.8. This is because there is the subj*sess variance component \
and the residual variance, too. You can review this in an anova table:

+---------------+-----------+----+-----------+-----+
|     Source    |     SS    | DF |     MS    | np2 |
+===============+===========+====+===========+=====+
|     subidr    | 20.008333 |  9 | 2.223148  | 1.0 |
+---------------+-----------+----+-----------+-----+
|      sess     | 29.216667 |  2 | 14.608333 | 1.0 |
+---------------+-----------+----+-----------+-----+
| subidr * sess | 22.616667 | 18 | 1.256481  | 1.0 |
+---------------+-----------+----+-----------+-----+
|    Residual   |   0.000000|  0 |    -      | -   |
+---------------+-----------+----+-----------+-----+


**icc_confint**

For each ICC estimate that can be requested, ICC(1), ICC(2,1) and ICC(3,1) and confidence interval \
is returned with each ICC estimate. The implementation for the confident interval is the same as in \
the the `pingouin <https://github.com/raphaelvallat/pingouin/blob/master/pingouin/reliability.py>`_ \
package in Python and the `ICC() from psych <https://search.r-project.org/CRAN/refmans/psych/html/ICC.html>`_ \
package in R.


**sumsq_icc**

Now that the internal calculations of the ICC have been reviewed, we can use the package to get the values of interest. \
The associated formulas for the ICC(1), ICC(2,1) and ICC(3,1) are described below.

The formula for ICC(1) is:

.. math::

    ICC(1) = \frac{MSb - MSw}{MS_b + (c-1)MS_w}


The formula for ICC(2,1) is:

.. math::

    ICC(2,1) = \frac{MSBtw - MSErr}{MSBtw + (c - 1) * MSErr + c * (MSc - MSErr) / n}

The formula for ICC(2,1) is:

.. math::

    ICC(3,1) = \frac{MSBtw - MSErr}{MSBtw + (c - 1) * MSErr}


Where:

- MSb: mean square between subjects
- MSw: mean square within subjects
- MSErr: mean squared residual error
- MSc: mean squared error of sessions
- c: is the number of sessions
- n: numbers of subjects


Hence, `sumsq_icc` can be used on a dataset with multiple subjects with 1+ measures occasions. We can calculate this ICC \
for the anagrams data used above. Note: the required inputs are a long dataframe, subject variable, \
session variable and the value scores variables that are contained in the long dataframe, plus the \
icc to return (options: icc_1, icc_2, icc_3; default: icc_3).

The `sumsq_icc` function will return five values: the ICC etimate, lower bound 95% confidence interval, \
upper bound 95% counfidence interval, mean square between subject variance, mean square within subject variance. \
This information will print to a terminal or can be saved to five variables:

.. code-block:: python

    # if you havent imported the package already
    from imgreliability import icc

    icc3, icc3_lb, icc3_up, icc3_msbs, icc3_msws = icc.sumsq_icc(df_long=a_ld,sub_var="subidr",
                                                    sess_var="sess",value_var="vals",icc_type="icc_3")

This will store the five associated values in the five variables:
    - `icc3`: ICC estimate
    - `icc3_lb`: 95% lower bound CI for ICC estimate
    - `icc3_lb`: 95% upper bound CI for ICC estimate
    - `icc3_msbs`: Mean Squared Between Subject Variance using for ICC estimate
    - `icc3_msws`: Mean Squared Within Subject Variance used for ICC estimate

brain_icc.py
------------

The `brain_icc.py` is, for a lack for better words, a big wrapper for for the `icc.py`. \
In short, the `voxelwise_icc` function within `brain_icc.py` calculates the ICC for 3D nifti brain images \
across subjects and sessions on a voxel-by-voxel basis. Here are the steps it uses:

    - The function takes in the paths to the 3D nifti brain images for each session, the path to the nifti mask object, and the ICC type to be calculated.
    - The function checks if there are the same number of files in session 1 and session 2 (e.g., paths_sess1, paths_sess2 + optional) and raises an error if they are of different length.
    - The function concatenates the 3D images into a 4D nifti image (4th dimension is subjects) using image.concat_imgs().
    - The function uses the provided nifti mask to mask the images using NiftiMasker.
    - It loops over the voxels in the `range(len(imgdata)` and creates a pandas DataFrame with the voxel values for each subject and session using sumsq_icc().
    - The function calculates and retuns to a list of five variables: ICC, lower and upper bounds of the ICC 95% confidence interval, mean square between subjects, and mean square within subjects using sumsq_icc().
    - The function then returns the five variables in the shape of the provided 3D volume using inverse_transform from NiftiMasker.

**voxelwise_icc**

As mentioned above, the `voxelwise_icc` calculates the ICC values for value in the 3D volumes. \
If we thing of an image as having the dimensions of [45, 45, 90], we can unravel it into a single vector \
for each subject that is 182,250 values long (the length in the voxelwise case is the number of voxels). \
The `voxelwise_icc` returns an equal size in length array that contains the ICC estimate for each voxels, \
between subjects across the measurement occasions. For example:

.. figure:: img_png/voxelwise_example.png
   :align: center
   :alt: Figure 1: HCP Left Hand (A) and Left Foot (B) Activation maps.
   :figclass: align-center

To use the `voxelwise_icc` function you just have to provide the following information:
    - paths_sess1: A list of paths to the Nifti z-stat, t-stat or beta maps for sess1 (or run 1)
    - paths_sess2: A list of paths to the Nifti z-stat, t-stat or beta maps for sess2 (or run 2)
    - paths_sess3: Optional; A list of paths to the Nifti z-stat, t-stat or beta maps for sess3 (or run 3)
    - mask: The Nifti binarized masks that will be used to mask the 3D volumes.
    - icc: The ICC estimate that will be calculated for each voxel. Options: `icc_1`, `icc_2`, `icc_3`. Default: `icc_3`

The function returns a 3D volume for:
    - ICC estimates
    - ICC lowerbound 95% CI
    - ICC upperbound 95% CI
    - Mean Squared Between Subject Variance
    - Mean Squared Within Subject Variance


Say we have stored paths to session 1 and session 2 in the following variables:

.. code-block:: python

    # session 1 paths
    scan1 = ["./scan1/sub-1_t-stat.nii.gz", "./scan1/sub-2_t-stat.nii.gz", "./scan1/sub-3_t-stat.nii.gz", "./scan1/sub-4_t-stat.nii.gz", "./scan1/sub-5_t-stat.nii.gz",
             "./scan1/sub-6_t-stat.nii.gz", "./scan1/sub-7_t-stat.nii.gz", "./scan1/sub-8_t-stat.nii.gz"]
    scan2 = ["./scan2/sub-1_t-stat.nii.gz", "./scan2/sub-2_t-stat.nii.gz", "./scan2/sub-3_t-stat.nii.gz", "./scan2/sub-4_t-stat.nii.gz", "./scan2/sub-5_t-stat.nii.gz",
             "./scan2/sub-6_t-stat.nii.gz", "./scan2/sub-7_t-stat.nii.gz", "./scan2/sub-8_t-stat.nii.gz"]

Next, you can call these images paths in the function and save the 3d volumes using:

.. code-block:: python

    from imgreliability import brain_icc

    icc_3d, icc_lb_3d, icc_ub_3d, icc_msbs_3d, icc_msws_3d = brain_icc.voxelwise_icc(paths_sess1 = scan1, paths_sess2 = scan2, mask = "./mask/brain_mask.nii.gz", icc = "icc_3")

This will return the associated nifti 3D volumes manipulated further, plotted or \
can be saved using nibabel:

.. code-block:: python

    import nibabel as nib
    nib.save(icc_3d, os.path.join('output_dir', 'file_name.nii.gz'))

