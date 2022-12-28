import os
import numpy as np
import pandas as pd
from nilearn import image
from itertools import combinations


def img_similarity(img1, img2, img_mask, thresh=None, similar_type='Dice'):
    """
    The img_similarity function takes in two images (3D), a binarized mask (3D), and simiarlity type.
    Based on the specified threshold and similarity type, the ratio of intersecting and union of voxels is calculated.
    The function returns a ratio of voxels that overlap between the two images

    :param img1:
    :param img2: list of paths to nii files
    :param img_mask: path to image mask for brain mask
    :param thresh: specify voxel threshold to use, if any, values >0. Default = 0
    :param similar_type: specify calculation, can be 'Dice' or 'Jaccard', Default = Dice
    :return: similarity coefficient
    """
    img_1 = image.load_img(img1)
    img_2 = image.load_img(img2)
    mask = image.load_img(img_mask)

    # test whether function type is of 'Dice' or 'Jaccard'
    if similar_type.casefold() != 'dice' and similar_type.casefold() != 'jaccard':
        raise Exception('StrError: [type] should be string "Dice" or "Jaccard". '
                        '\n Value provided: {}'.format(similar_type))

    if img_1.shape != img_1.shape:
        raise Exception('ValueError: Image shapes are not equal. \n '
                        'Image shapes provided, image 1 {} and image 2 {}.'.format(img_1.shape, img_2.shape))

    if thresh is None:
        # mask input images by provided mask. Returns numpy 1D data array
        img_1_masked = masking.apply_mask(img_1, mask)
        img_2_masked = masking.apply_mask(img_2, mask)

        # calc intersection/union
        intersect = np.logical_and(img_1_masked, img_2_masked)
        union = np.logical_or(img_1_masked, img_2_masked)

        # using np.finfo().eps to avoid non-zero division errors
        dice_coeff = (intersect.sum()) / (float(union.sum()) + np.finfo(float).eps)

    elif thresh > 0:
        img1_thresh = image.threshold_img(img1, threshold=thresh, two_sided=True)
        img2_thresh = image.threshold_img(img2, threshold=thresh, two_sided=True)

        img_1_masked = masking.apply_mask(img1_thresh, mask)
        img_2_masked = masking.apply_mask(img2_thresh, mask)

        img_1_maskedthr = img_1_masked > thresh
        img_2_maskedthr = img_2_masked > thresh

        intersect = np.logical_and(img_1_maskedthr, img_2_maskedthr)
        union = np.logical_or(img_1_maskedthr, img_2_maskedthr)
        dice_coeff = intersect.sum() / (float(union.sum()) + np.finfo(float).eps)

    elif thresh < 0:
        img1_thresh = image.threshold_img(img1, threshold=thresh, two_sided=True)
        img2_thresh = image.threshold_img(img2, threshold=thresh, two_sided=True)

        img_1_masked = masking.apply_mask(img1_thresh, mask)
        img_2_masked = masking.apply_mask(img2_thresh, mask)

        img_1_maskedthr = img_1_masked < thresh
        img_2_maskedthr = img_2_masked < thresh

        intersect = np.logical_and(img_1_maskedthr, img_2_maskedthr)
        union = np.logical_or(img_1_maskedthr, img_2_maskedthr)
        dice_coeff = intersect.sum() / (float(union.sum()) + np.finfo(float).eps)

    if similar_type.casefold() == 'dice':
        coef = dice_coeff

    elif similar_type.casefold() == 'jaccard':
        jaccards = dice_coeff / (2 - dice_coeff)
        coef = jaccards

    return coef


def permute_images(nii_list, img_mask, thresh=1.5, similar_type='Dice'):
    """
    This permutation takes in a list of paths to Nifti images and creates a comparsion that covers all possible
    combinations. For each combination, it calculates the specified similarity and
    returns the coefficients & string combo.

    :param nii_list: list of paths to NII files
    :param img_mask: img_mask: path to image mask for brain mask
    :param thresh: threshold to use on NII files, default 1.5
    :param similar_type: type of similarity calc, Dice or Jaccards, default Dice
    :return: returns similarity coefficient & labels in pandas dataframe
    """
    # test whether function type is of 'Dice' or 'Jaccard', case insensitive
    if similar_type.casefold() != 'dice' and similar_type.casefold() != 'jaccard':
        raise Exception('StrError: [type] should be string "Dice" or "Jaccard". '
                        '\n Value provided for {}'.format(similar_type))

    var_permutes = list(combinations(nii_list, 2))
    coef_df = pd.DataFrame(columns=['similar_coef', 'img_labs'])

    for r in var_permutes:
        # select basename of file name(s)
        path1 = os.path.basename(os.path.normpath(r[0]))
        path2 = os.path.basename(os.path.normpath(r[1]))
        # calculate simiarlity & create label
        val = img_similarity(img1=r[0], img2=r[1], img_mask=img_mask,
                             thresh=thresh, similar_type=similar_type)
        label = [path1, path2]
        # for each permutation, save value + label to pandas df
        similarity_data = pd.DataFrame(np.column_stack((val, " ~ ".join(label))),
                                       columns=['similar_coef', 'img_labs'])
        coef_df = pd.concat([coef_df, similarity_data], axis=0, ignore_index=True)

    return coef_df

def sumsq_total(df_long):
    """
    calculates the sum of square total
    the difference between each value and the global mean
    :param df_long:
    :return:
    """
    np.square(
        np.subtract(df_long["vals"], df_long["vals"].mean())
    ).sum()

def sumsq_within(df_long, n):
    """
    calculates the sum of squared Intra-subj variance,
    the average session value subtracted from overall avg of values
    :param df_long: long df
    :param n: sample n
    :return: returns sumsqured within
    """
    return np.multiply(
        np.square(
            np.subtract(df_long['vals'].mean(),
                        df_long[['sess', 'vals']].groupby(by='sess')['vals'].mean()
                        )),
        n
    ).sum()

def sumsq_btwn(df_long, c):
    """
    calculates the sum of squared between-subj variance,
    the average subject value subtracted from overall avg of values
    :param df_long: long df
    :param n: sample n
    :return: returns sumsqured within
    """
    return np.multiply(
        np.square(
            np.subtract(df_long['vals'].mean(),
                        df_long[['sub_var', 'vals']].groupby(by=sub_var)['vals'].mean()
                        )),
        c
    ).sum()

def calc_icc(wide, sub_var, sess_vars, icc_type='icc_2'):
    """
    This ICC calculation employs the ANOVA technique.
    It converts a wide data.frame into a long format, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.


    :param wide: Data of subjects & sessions, wide format.
    :param sub_var: list of variables in dataframe that subject identifying variable
    :param sess_vars: list of in dataframe that are repeat session variables
    :param icc_type: default is ICC(2,1), alternative is ICC(1,1) via icc_1 or ICC(3,1) via icc_3
    :return: ICC calculation
    """
    df_long = pd.melt(wide,
                      id_vars=sub_var,
                      value_vars=sess_vars,
                      var_name='sess',
                      value_name='vals')

    # Calc DF
    [n, c] = wide.drop([sub_var], axis=1).shape
    DF_n = n - 1
    DF_c = c - 1
    DF_r = (n - 1) * (c - 1)

    # Sum of Square Vals
    # sum of squared total
    SS_T = sumsq_total(df_long)

    # the sum of squared inter-subj variance (c = sessions)
    SS_R = sumsq_btwn(df_long, c)

    # the sum of squared Intra-subj variance (n = sample of subjects)
    SS_C = sumsq_c(df_long, n)

    # Sum Square Errors
    SSE = SS_T - SS_R - SS_C

    # Sum square withins subj err
    SSW = SS_C + SSE

    # Mean Squared Values
    MSR = SS_R / (DF_n)
    MSC = SS_C / (DF_c)
    MSE = SSE / (DF_r)
    MSW = SSW / (n * (DF_c))

    if icc_type == 'icc_2':
        # ICC(2,1)
        ICC_est = (MSR - MSE) / (MSR + (DF_c) * MSE + (c) * (MSC - MSE) / n)

    elif icc_type == 'icc_1':
        # ICC(1), Model 1
        ICC_est = (MSR - MSW) / (MSR + (DF_c * MSW))
    elif icc_type == 'icc_3':
        # ICC(3,1)
        ICC_est = (MSR - MSE) / (MSR + (DF_c) * MSE)
    else:
        raise Exception('[icc_type] should be of icc_1, icc_2 or icc_3. \n Value type provided: {}'.format(icc_type))

    return ICC_est


def mri_voxel_icc(sess1, sess2, img_mask, sess3=None, icc='icc_2'):
    """
    mri_voxel_icc: calculates the ICC by voxel for specified input files.
    The path to the subject's data should be provided as a list for each session, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz", "./ses1/sub-00_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz", "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    Inter-subject variance would be: between subjects in session 1 & between subjects in session 2
    Intra-subject variance would be: within subject across session 1 and session 2.

    :param sess1: paths to session 2 nii MNI files
    :param sess2: paths to session 2 nii MNI files
    :param sess3: If there are more than 3 sessions, paths to session 3 nii MNI files
    :param mask: path to nii MNI path object
    :param icc: provide icc type, default is icc_2, options: icc_1, icc_2, icc_3
    :return: returns 3D shaped array of ICCs in shape of provided 3D  mask
    """

    # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
    sess1_dat = image.concat_imgs(sess1)
    sess2_dat = image.concat_imgs(sess2)

    # get subj details per session list to confirm size is equal
    subjs = sess1_dat.shape[-1]
    sub_n = np.array(np.arange(start=0, stop=subjs, step=1))

    # mask & convert session vols to 2D (position 1 = image number, position 2 = voxel number)
    sess1_img_2d = masking.apply_mask(imgs=sess1_dat, mask_img=img_mask)
    sess2_img_2d = masking.apply_mask(imgs=sess2_dat, mask_img=img_mask)

    ICCs = []

    if sess3 is None:
        for i in range(len(sess1_img_2d.T)):
            # sub sample i voxel for all subjects for each session
            sess1_voxs = sess1_img_2d[:, i]
            sess2_voxs = sess2_img_2d[:, i]

            # stack columns to create np array that includes voxels and sub labels
            np_voxdata = np.column_stack((sub_n, sess1_voxs, sess2_voxs))

            # create dataframe that is then used with ICC function to calculate ICC
            ICC_data = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2"])
            ICCs.append(calc_icc(ICC_data, "subj", ["sess1", "sess2"], icc_type=icc))

        # using unmask to reshape the 1D voxels back to 3D specified mask
        ICC_array = np.array(ICCs)
        ICC_reshaped = masking.unmask(X=ICC_array, mask_img=mask)

    elif sess3 is not None:
        # get data for this new 4D session volume
        sess3_dat = image.concat_imgs(sess3)

        # convert session vols to 2D
        sess3_img_2d = masking.apply_mask(imgs=sess3_dat, mask_img=mask)

        for i in range(len(sess1_img_2d.T)):
            # sub sample i voxel for all subjects for each session
            sess1_voxs = sess1_img_2d[:, i]
            sess2_voxs = sess2_img_2d[:, i]
            sess3_voxs = sess3_img_2d[:, i]

            # stack columns to create np array that includes voxels and sub labels
            np_voxdata = np.column_stack((sub_n, sess1_voxs, sess2_voxs, sess3_voxs))

            # create dataframe that is then used with ICC function to calculate ICC
            ICC_data = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2", "sess3"])
            ICCs.append(calc_icc(ICC_data, "subj", ["sess1", "sess2", "sess3"], icc_type=icc))

        # using unmask to reshape the 1D voxels back to 3D specified mask
        ICC_array = np.array(ICCs)
        ICC_reshaped = masking.unmask(X=ICC_array, mask_img=mask)

    return ICC_reshaped
