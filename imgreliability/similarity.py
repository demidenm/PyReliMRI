import os
import numpy as np
import pandas as pd
from nilearn import image
from itertools import combinations
from nilearn.maskers import NiftiMasker


def image_similarity(imgfile1: str, imgfile2: str,
                     mask: str = None, thresh: float = None,
                     similarity_type: str = 'Dice') -> float:
    """ The img_similarity function takes in two images (3D), a binarized mask (3D), and simiarlity type.
    Based on the specified threshold and similarity type, the ratio of intersecting and union of voxels is calculated.
    The function returns a ratio of voxels that overlap between the two images
    :param imgfile1: nii path to first image
    :param imgfile2: nii path to second image
    :param mask: path to image mask for voxel selection
    :param thresh: specify voxel threshold to use, if any, values >0. Default = 0
    :param similarity_type: specify calculation, can be Dice or Jaccards, Default = Dice
    :return: similarity coefficient
    """
    assert similarity_type.casefold() in ['dice', 'jaccard'], 'similarity_type must be "Dice" or "Jaccard". ' \
                                                              'Provided: {}"'.format(similarity_type)

    # load list of images
    imagefiles = [imgfile1, imgfile2]
    img = [image.load_img(i) for i in imagefiles]

    assert img[0].shape == img[1].shape, 'images of different shape, ' \
                                         'image 1 {} and image 2 {}'.format(img[0].shape, img[1].shape)

    # mask image
    masker = NiftiMasker(mask_img=mask)
    imgdata = masker.fit_transform(img)

    # threshold image, compatible for positive & negative values
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


def permute_images(nii_filelist: list, mask: str,
                   thresh: float = None, similarity_type: str = 'Dice'):
    """This permutation takes in a list of paths to Nifti images and creates a comparsion that covers all possible
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