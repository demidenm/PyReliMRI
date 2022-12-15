import os
import numpy as np
import pandas as pd
from nilearn import image
from itertools import combinations

# some general/overall comments:
# first - nice job overall! 

# generally, it would be nice for all of your functions to have the same interface
# e.g. they all take a list of images and a mask, and return a similarity coefficient
# I would be inclined to follow the scikit-learn interface model, though it requires
# writing classes rather than plain functions.
# I would get it working with plain functions first, then refactor to classes

# It would be nice to have a description of the intended purpose of each function
# within the docstring

# in general, putting indices into variable names is a bad practice
# it would be better to put the images into a list

# use a more descriptive function name
# I am not actually sure 
# type is a reserved term in python, and you should also be more specific about
# what the parameter means
def similarity(img1, img2, thresh=0, type='Dice'):
    """

    :param imgs: list of paths to nii files
    :param mask: path to image mask for voxel selection
    :param thresh: specify voxel threshold to use, if any, values >0. Default = 0
    :param type: specify calculation, can be Dice or Jaccards, Default = Dice
    :return: similarity coefficient
    """
    # there was a lot of repeated code here

    # probably best to set thresh to None in the function definition
    if thresh == 0:
        img_1st = image.load_img(img1)
        img_2nd = image.load_img(img2)

        # everything from here down in the if statement is repeated in the else statement
        # so you can just do this once
        img1_dat = image.get_data(img_1st)
        img2_dat = image.get_data(img_2nd)

        if img1_dat.shape != img1_dat.shape:
            # if you are going to raise an exception it should be as specific as possible,
            # e.g. ValueError
            raise Exception('Image shapes are no equal. Please check shape of input files.')
        else:
            # this section doesn't need to go into an else statement
            # you can just do it after the if statement
            # since the if statement will exit the function if it is true
            intersect = np.logical_and(img1_dat, img2_dat)
            union = np.logical_or(img1_dat, img2_dat)
            # what is the intended purpose of rounding here?
            dice_coeff = round((intersect.sum()) / (float(union.sum()) + np.finfo(float).eps), 3)

    # what if I wanted to use a negative threshold? this method wouldn't work
    elif thresh > 0:
        img1_thresh = image.threshold_img(img1, threshold=thresh)
        img2_thresh = image.threshold_img(img2, threshold=thresh)
        # should check to make sure that the thresholded images are not empty

        img1_dat = image.get_data(img1_thresh)
        img2_dat = image.get_data(img2_thresh)

        # it would be more pythonic to do this in a try/catch block
        if img1_dat.shape == img2_dat.shape:
            intersect = np.logical_and(img1_dat, img2_dat)
            union = np.logical_or(img1_dat, img2_dat)
            dice_coeff = round(intersect.sum() / (float(union.sum()) + np.finfo(float).eps), 3)

        else:
            # would be nice to actually print the shapes of the mismatching images
            raise Exception('Image shapes are no equal. Please check shape of input files.')
    else:
        # what I would do is have a test at the very top for the value of the threshold (if needed)
        raise Exception('[thresh]should be integer >0. \n Value provided: {}'.format(thresh))

    if type == 'Dice':
        coef = dice_coeff
    # it is usually called the Jaccard coefficient, not Jaccard's coefficient
    elif type == 'Jaccards':
        jaccards = dice_coeff / (2 - dice_coeff)
        coef = jaccards
    else:
        # what I would do is have a test at the very top for the value of the coefficient type
        raise Exception('[type] should be string Dice or Jaccards. \n Value provided: {}'.format(type))

    return coef

# the naming here doesn't seem exactly right, since it implies that you are 
# permuting the similarity values when you are actually permuting the images
def permute_similarity(nii_list, thresh=1.5, type='Dice'):
    """
    :param nii_list: list of paths to NII files
    :param tresh: threshold to use on NII files, default 1.5
    :param type: type of similarity calc, Dice or Jaccards, default Dice
    :return: returns similarity coefficient for all nii permutations
    """
    var_permutes = list(combinations(nii_list, 2))

    # I don't love this way of storing the results since it somewhat separates
    # the data from the labels. I would use a pandas dataframe or dictionary instead
    coef_dat = []
    coef_label = []

    for r in var_permutes:
        val = similarity(r[0], r[1], thresh=thresh, type=type)
        coef_dat.append([val])

        path1 = os.path.basename(os.path.normpath(r[0]))
        path2 = os.path.basename(os.path.normpath(r[1]))
        coef_label.append([path1, path2])

    return coef_dat,coef_label

# shoudl follow python function naming conventions by using camel case: https://peps.python.org/pep-0008/#function-and-variable-names

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

    # for the sake of modularity I would separate out the computatoin of the
    # sums of squares into a separate function
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
        # I would put this at the top of the function
        raise Exception('[icc_type] should be of icc_1, icc_2 or icc_3. \n Value type provided: {}'.format(icc_type))

    return ICC_est


# I don't think you need this, as you could use nilearn.maskers.niftimasker
# which would also deal with masking and other preprocessing operations
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

# the naming here is a bit unclear - should be clearer on exactly what this function does
# what is the purpose of sess3? 
# if sess3 is not required then I would define it as None
# then you wouldn't actually need the n_sessions variable, you could just check if sess3 is None
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

    # there is a lot of repeated code here
    # you should figure out what exactly is unique to this case
    # the code looks similar enough between the 2 and 3 session cases that you could 
    # probably combine them

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
        # doesn't np.arange already return an array?
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