import os
import numpy as np
import pandas as pd
from nilearn import image
from itertools import combinations


def similarity(img1, img2, thresh=0, type='Dice'):
    """

    :param imgs: list of paths to nii files
    :param thresh: specify voxel threshold to use, if any, values >0. Default = 0
    :param type: specify calculation, can be Dice or Jaccards, Default = Dice
    :return: similarity coefficient
    """

    if thresh == 0:
        img_1st = image.load_img(img1)
        img_2nd = image.load_img(img2)

        img1_dat = image.get_data(img_1st)
        img2_dat = image.get_data(img_2nd)

        if img1_dat.shape != img1_dat.shape:
            raise Exception('Image shapes are no equal. Please check shape of input files.')
        else:
            intersect = np.logical_and(img1_dat, img2_dat)
            union = np.logical_or(img1_dat, img2_dat)
            dice_coeff = round((intersect.sum()) / (float(union.sum()) + np.finfo(float).eps), 3)

    elif thresh > 0:
        img1_thresh = image.threshold_img(img1, threshold=thresh)
        img2_thresh = image.threshold_img(img2, threshold=thresh)
        img1_dat = image.get_data(img1_thresh)
        img2_dat = image.get_data(img2_thresh)

        if img1_dat.shape == img2_dat.shape:
            intersect = np.logical_and(img1_dat, img2_dat)
            union = np.logical_or(img1_dat, img2_dat)
            dice_coeff = round(intersect.sum() / (float(union.sum()) + np.finfo(float).eps), 3)

        else:
            raise Exception('Image shapes are no equal. Please check shape of input files.')
    else:
        raise Exception('[thresh]should be integer >0. \n Value provided: {}'.format(thresh))

    if type == 'Dice':
        coef = dice_coeff
    elif type == 'Jaccards':
        jaccards = dice_coeff / (2 - dice_coeff)
        coef = jaccards
    else:
        raise Exception('[type] should be string Dice or Jaccards. \n Value provided: {}'.format(type))

    return coef


def permute_similarity(nii_list, thresh=1.5, type='Dice'):
    """
    :param nii_list: list of paths to NII files
    :param tresh: threshold to use on NII files, default 1.5
    :param type: type of similarity calc, Dice or Jaccards, default Dice
    :return: returns similarity coefficient for all nii permutations
    """
    var_permutes = list(combinations(nii_list, 2))

    coef_dat = []
    coef_label = []

    for r in var_permutes:
        val = similarity(r[0], r[1], thresh=thresh, type=type)
        coef_dat.append([val])

        path1 = os.path.basename(os.path.normpath(r[0]))
        path2 = os.path.basename(os.path.normpath(r[1]))
        coef_label.append([path1, path2])

    return coef_dat,coef_label


def Calc_icc(wide, sub_var, sess_vars, icc_type='icc_2'):
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
    # sum of squared total, the difference between each value & the overall mean
    SS_T = np.square(
        np.subtract(df_long["vals"], df_long["vals"].mean())
    ).sum()

    # the sum of squared inter-subj variance, the average subject value subtracted from overall avg of values
    SS_R = np.multiply(
        np.square(
            np.subtract(df_long['vals'].mean(),
                        df_long[[sub_var, 'vals']].groupby(by=sub_var)['vals'].mean()
                        )),
        c
    ).sum()

    # the sum of squared Intra-subj variance, the average session value subtracted from overall avg of values
    SS_C = np.multiply(
        np.square(
            np.subtract(df_long['vals'].mean(),
                        df_long[['sess', 'vals']].groupby(by='sess')['vals'].mean()
                        )),
        n
    ).sum()

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


def convrt_4Dto2D(img):
    """
    This is used to conver the 4D to 2D image
    :param img: provide a 4D numpy array (i.e., converted 4D nifti)
    :return: reshape 2D volume, 1-3 collapsed to dim 1 and 4 is now dim 2
    """
    # Get length of concatenated images & create array of pseudo subject labels 1 - N sub imgs
    subjs = img.shape[-1]

    # Get the 3D shape of the images, then calculate the number of voxels
    shape_3d = img.shape[:-1]
    vox_n = np.prod(shape_3d)

    # reshape the session images into voxels (length) by subject (columns)
    img_vox_by_sub = np.reshape(img, (vox_n, img.shape[-1]))

    return img_vox_by_sub


def MRI_ICCs(sess1, sess2, sess3, n_sessions=2, icc='icc_2'):
    """
    MRI_ICCs calculates the ICC for specified input files. The path to the subject's data should be provided as a list
    for each session, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz", "./ses1/sub-00_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz", "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    Inter-subject variance would be: between subjects in session 1 & between subjects in session 2
    Intra-subject variance would be: within subject across session 1 and session 2.

    :param sess1:
    :param sess2:
    :param sess3:
    :param sess4:
    :param n_sessions:
    :param icc: provide icc type, default is icc_2, options: icc_1, icc_2, icc_3
    :return: returns list of calculated ICCs in shape of 3D img
    """

    if n_sessions == 2:
        # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
        # get data for this new 4D volume
        sess1_dat = image.concat_imgs(sess1)
        sess1_img = image.get_data(sess1_dat)
        sess2_dat = image.concat_imgs(sess2)
        sess2_img = image.get_data(sess2_dat)

        # get subj details per session list to confirm size is equal
        subjs = sess1.shape[-1]
        shape3D_img = sess1_img.shape[:-1]
        sub_n = np.array(np.arange(start=0, stop=subjs, step=1))

        # convert session vols to 2D
        sess1_img_2d = convrt_4Dto2D(sess1_img)
        sess2_img_2d = convrt_4Dto2D(sess2_img)

        for i in range(len(sess1_img_2d)):
            # sub sample i voxel for all subjects for each session
            sess1_voxs = sess1_img_2d[i, :]
            sess2_voxs = sess2_img_2d[i, :]

            # stack columns to create np array that includes voxels and sub labels
            np_voxdata = np.column_stack((sub_n, sess1_voxs, sess2_voxs))

            # create dataframe that is then used with ICC function to calculate ICC
            ICC_data = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2"])
            ICCs.append(Calc_icc(ICC_data, "subj", ["sess1", "sess2"], icc_type=icc))

    if n_sessions == 3:
        # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
        # get data for this new 4D volume
        sess1_dat = image.concat_imgs(sess1)
        sess1_img = image.get_data(sess1_dat)
        sess2_dat = image.concat_imgs(sess2)
        sess2_img = image.get_data(sess2_dat)
        sess3_dat = image.concat_imgs(sess3)
        sess3_img = image.get_data(sess3_dat)

        # get subj details per session list to confirm size is equal
        subjs = sess1.shape[-1]
        shape3D_img = sess1_img.shape[:-1]
        sub_n = np.array(np.arange(start=0, stop=subjs, step=1))

        # convert session vols to 2D
        sess1_img_2d = convrt_4Dto2D(sess1_img)
        sess2_img_2d = convrt_4Dto2D(sess2_img)
        sess3_img_2d = convrt_4Dto2D(sess3_img)

        for i in range(len(sess1_img_2d)):
            # sub sample i voxel for all subjects for each session
            sess1_voxs = sess1_img_2d[i, :]
            sess2_voxs = sess2_img_2d[i, :]
            sess3_voxs = sess3_img_2d[i, :]

            # stack columns to create np array that includes voxels and sub labels
            np_voxdata = np.column_stack((sub_n, sess1_voxs, sess2_voxs, sess3_voxs))

            # create dataframe that is then used with ICC function to calculate ICC
            ICC_data = pd.DataFrame(data=np_voxdata, columns=["subj", "sess1", "sess2", "sess3"])
            ICCs.append(Calc_icc(ICC_data, "subj", ["sess1", "sess2", "sess3"], icc_type=icc))

    return ICCs.reshape(shape3D_img)