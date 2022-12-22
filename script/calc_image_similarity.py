import os
import numpy as np
import pandas as pd
from nilearn import image
from itertools import combinations
from nilearn.maskers import NiftiMasker

def image_similarity(imgfile1, imgfile2, mask=None, thresh=None, similarity_type='Dice'):
    """
    :param imgfile1: nii path to first image
    :param imgfile2: nii path to second image
    :param mask: path to image mask for voxel selection
    :param thresh: specify voxel threshold to use, if any, values >0. Default = 0
    :param similarity_type: specify calculation, can be Dice or Jaccards, Default = Dice
    :return: similarity coefficient
    """
    assert similarity_type.casefold() in ['dice','jaccard'], 'similarity_type must be "Dice" or "Jaccard". ' \
                                                             'Provided: {}"'.format(similarity_type)

    # load list of images
    imagefiles = [imgfile1, imgfile2]
    img = [image.load_img(i) for i in imagefiles]

    assert img[0].shape == img[1].shape, 'images of different shape, ' \
                                         'image 1 {} and image 2 {}'.format(img[0].shape,img[1].shape)

    # mask image
    masker = NiftiMasker(mask_img=mask)
    imgdata = masker.fit_transform(img)

    # threshold image, make compatible for positive & negative values
    # (i.e., some may want similarity in (de)activation)
    if thresh is not None:
        if thresh > 0:
            imgdata = imgdata > thresh

        elif thresh < 0:
            imgdata = imgdata < thresh

    intersect = np.logical_and(imgdata[0, :], imgdata[1, :])
    union = np.logical_or(imgdata[0, :], imgdata[1, :])
    dice_coeff = (intersect.sum()) / (float(union.sum()) + np.finfo(float).eps)

    return dice_coeff if similarity_type.casefold() == 'dice' else dice_coeff / (2 - dice_coeff)


def permute_images(nii_filelist, mask, thresh=None, similarity_type='Dice'):
    """
    This permutation takes in a list of paths to Nifti images and creates a comparsion that covers all possible
    combinations. For each combination, it calculates the specified similarity and
    returns the coefficients & string combo.

    :param nii_filelist: list of paths to NII files
    :param mask: path to image mask for brain mask
    :param thresh: threshold to use on NII files, default 1.5
    :param similarity_type: type of similarity calc, Dice or Jaccards, default Dice
    :return: returns similarity coefficient & labels in pandas dataframe
    """
    # test whether function type is of 'Dice' or 'Jaccard', case insensitive
    assert similarity_type.casefold() in ['dice', 'jaccard'], 'similarity_type must be "Dice" or "Jaccard", ' \
                                                   '{} entered'.format(similarity_type)

    var_permutes = list(combinations(nii_filelist, 2))
    coef_df = pd.DataFrame(columns=['similar_coef', 'image_labels'])

    for r in var_permutes:
        # select basename of file name(s)
        path = [os.path.basename(i) for i in r]
        # calculate simiarlity
        val = image_similarity(imgfile1=r[0], imgfile2=r[1], mask=mask, thresh=thresh, similarity_type=similarity_type)

        # for each permutation, save value + label to pandas df
        similarity_data = pd.DataFrame(np.column_stack((val, " ~ ".join([path[0], path[1]]))),
                                       columns=['similar_coef', 'image_labels'])
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
                        df_long[[sub_var, 'vals']].groupby(by=sub_var)['vals'].mean()
                        )),
        c
    ).sum()


def calculate_icc(df_wide, sub_var, sess_vars, icc_type='icc_3'):
    """
    This ICC calculation employs the ANOVA technique.
    It converts a wide data.frame into a long format, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.


    :param df_wide: Data of subjects & sessions, wide format.
    :param sub_var: list of variables in dataframe that subject identifying variable
    :param sess_vars: list of in dataframe that are repeat session variables
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: ICC calculation
    """
    assert icc_type in ['icc_1', 'icc_2','icc_3'], 'ICC type should be icc_1, icc_2,icc_3, ' \
                                                   '{} entered'.format(icc_type)

    df_long = pd.melt(df_wide,
                      id_vars=sub_var,
                      value_vars=sess_vars,
                      var_name='sess',
                      value_name='vals')

    # Calc degrees of freedom
    [n, c] = df_wide.drop([sub_var], axis=1).shape
    DF_n = n - 1
    DF_c = c - 1
    DF_r = (n - 1) * (c - 1)

    # Calculating different sum of squared values
    # sum of squared total
    SS_T = sumsq_total(df_long)

    # the sum of squared inter-subj variance (c = sessions)
    SS_R = sumsq_btwn(df_long, c)

    # the sum of squared Intra-subj variance (n = sample of subjects)
    SS_C = sumsq_within(df_long, n)

    # Sum Square Errors
    SSE = SS_T - SS_R - SS_C

    # Sum square withins subj err
    SSW = SS_C + SSE

    # Mean Squared Values
    MSR = SS_R / (DF_n)
    MSC = SS_C / (DF_c)
    MSE = SSE / (DF_r)
    MSW = SSW / (n * (DF_c))

    if icc_type == 'icc_1':
        # ICC(1), Model 1
        ICC_est = (MSR - MSW) / (MSR + (DF_c * MSW))

    elif icc_type == 'icc_2':
        # ICC(2,1)
        ICC_est = (MSR - MSE) / (MSR + (DF_c) * MSE + (c) * (MSC - MSE) / n)

    elif icc_type == 'icc_3':
        # ICC(3,1)
        ICC_est = (MSR - MSE) / (MSR + (DF_c) * MSE)

    return ICC_est


def mri_voxel_icc(paths_sess1, paths_sess2, mask, paths_sess3=None, icc='icc_3'):
    """
    mri_voxel_icc: calculates the ICC by voxel for specified input files.
    The path to the subject's data should be provided as a list for each session, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz", "./ses1/sub-00_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz", "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    Inter-subject variance would be: between subjects in session 1 & between subjects in session 2
    Intra-subject variance would be: within subject across session 1 and session 2.

    :param paths_sess1: paths to session 2 nii MNI files
    :param paths_sess2: paths to session 2 nii MNI files
    :param paths_sess3: If there are more than 3 sessions, paths to session 3 nii MNI files
    :param mask: path to nii MNI path object
    :param icc: provide icc type, default is icc_3, options: icc_1, icc_2, icc_3
    :return: returns 3D shaped array of ICCs in shape of provided 3D  mask
    """

    assert len(paths_sess1) == len(paths_sess2), 'sessions lists do not match, ' \
                                                 'session 1 length: {} and session 2 length: {}'.format(
        len(paths_sess1), len(paths_sess2))

    # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
    session_files = [paths_sess1, paths_sess2] if paths_sess3 is None else [paths_sess1, paths_sess2, paths_sess3]
    session_data = [image.concat_imgs(i) for i in session_files]

    # mask images
    masker = NiftiMasker(mask_img=mask)
    imgdata = [masker.fit_transform(i) for i in session_data]

    # get subj details per session to use in pandas df
    subjs = imgdata[0].shape[:-1]
    sub_n = np.array(np.arange(start=0, stop=subjs[0], step=1))

    ICC = []

    if paths_sess3 is None:
        for v in range(len(imgdata[0].T)):
            # sub sample v voxel for all subjects for each session. Stack columns to create np array
            #  that includes voxels and sub labels for session 1 [0] & session 2 [1]
            np_voxdata = np.column_stack((sub_n, imgdata[0][:, v], imgdata[1][:, v]))

            # create dataframe that is then used with ICC function to calculate specified ICC
            vox_pd = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2"])
            ICC.append(calculate_icc(vox_pd, "subj", ["sess1", "sess2"], icc_type=icc))

        # using unmask to reshape the 1D voxels back to 3D specified mask
        icc_array = np.array(ICC)
        icc_brain = masker.inverse_transform(icc_array)

    elif paths_sess3 is not None:
        for v in range(len(imgdata[0].T)):
            # sub sample v voxel for all subjects for each session. Stack columns to create np array
            #  that includes voxels and sub labels for session 1 [0] & session 2 [1], session [3]
            np_voxdata = np.column_stack((sub_n, imgdata[0][:, v], imgdata[1][:, v], imgdata[2][:, v]))

            # create dataframe that is then used with ICC function to calculate specified ICC
            vox_pd = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2", "sess3"])
            ICC.append(calculate_icc(vox_pd, "subj", ["sess1", "sess2", "sess3"], icc_type=icc))

        # using unmask to reshape the 1D voxels back to 3D specified mask
        icc_array = np.array(ICC)
        icc_brain = masker.inverse_transform(icc_array)

    return icc_brain