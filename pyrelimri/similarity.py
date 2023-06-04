import os
import warnings
from pandas import concat, DataFrame
import numpy as np
from nilearn import image
from itertools import combinations
from nilearn.maskers import NiftiMasker
from scipy.stats import spearmanr
from pyrelimri.tetrachoric_correlation import tetrachoric_corr as tet_corr


def image_similarity(imgfile1: str, imgfile2: str,
                     mask: str = None, thresh: float = None,
                     similarity_type: str = 'dice') -> float:
    """
    The image_similarity function takes in two images (3D), a binarized mask (3D), and simiarlity type.
    Based on the specified threshold and similarity type, the ratio of intersecting and union of voxels is calculated.
    The function returns a ratio of voxels that overlap between the two images
    :param imgfile1: nii path to first image
    :param imgfile2: nii path to second image
    :param mask: path to image mask for voxel selection
    :param thresh: specify voxel threshold to use, if any, values >0 or < 0. Default = 0
    :param similarity_type: specify calculation, can be Dice, Jaccards, Tetrachoric, or Spearman, Default = Dice
    :return: similarity coefficient
    """
    assert similarity_type.casefold() in ['dice', 'jaccard',
                                          'tetrachoric', 'spearman'], 'similarity_type must be ' \
                                                                      '"Dice", "Jaccard", "Tetrachoric" or ' \
                                                                      '"Spearman". Provided: {}"'.format(
        similarity_type)

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
    if thresh is not None and similarity_type.casefold() != 'spearman':
        if thresh > 0:
            imgdata = imgdata > thresh

        elif thresh < 0:
            imgdata = imgdata < thresh

    if similarity_type.casefold() in ['dice', 'jaccard']:
        # Intersection of images
        intersect = np.logical_and(imgdata[0, :], imgdata[1, :])
        union = np.logical_or(imgdata[0, :], imgdata[1, :])
        dice_coeff = (intersect.sum()) / (float(union.sum()) + np.finfo(float).eps)
        if similarity_type.casefold() == 'dice':
            coeff = dice_coeff
        else:
            coeff = dice_coeff / (2 - dice_coeff)
    elif similarity_type.casefold() == 'tetrachoric':
        warnings.filterwarnings('ignore')
        coeff = tet_corr(vec1=imgdata[0, :], vec2=imgdata[1, :])

    else:
        if thresh is not None:
            raise ValueError(f"It is recommended to perform Spearman rank on images that are not thresholded."
                             f" Currently threshold is set to: {thresh}. Recommended 'None'.")
        else:
            coeff = spearmanr(a=imgdata[0, :], b=imgdata[1, :])[0]

    return coeff


def pairwise_similarity(nii_filelist: list, mask: str = None,
                   thresh: float = None, similarity_type: str = 'Dice') -> DataFrame:
    """This pairwise comparison takes in a list of paths to Nifti images and creates
    a comparsion that covers all possible combinations. For each combination, it calculates
    the specified similarity and returns the coefficients & string combo.

    :param nii_filelist: list of paths to NII files
    :param mask: path to image mask for brain mask
    :param thresh: specify voxel threshold to use, if any, values >0 or < 0. Default = 0
    :param similarity_type: type of similarity calc, Dice or Jaccards, Tetrachoric or Spearman Correlation, default Dice
    :return: returns similarity coefficient & labels in pandas dataframe
    """
    # test whether function type is of 'Dice' or 'Jaccard', case insensitive
    assert similarity_type.casefold() in ['dice', 'jaccard',
                                          'tetrachoric','spearman'], 'similarity_type must be ' \
                                                                     '"Dice", "Jaccard", "Tetrachoric" or ' \
                                                                     '"Spearman". Provided: {}"'.format(similarity_type)

    var_pairs = list(combinations(nii_filelist, 2))
    coef_df = DataFrame(columns=['similar_coef', 'image_labels'])

    for img_comb in var_pairs:
        # select basename of file name(s)
        path = [os.path.basename(i) for i in img_comb]
        # calculate simiarlity
        val = image_similarity(imgfile1=img_comb[0], imgfile2=img_comb[1], mask=mask,
                               thresh=thresh, similarity_type=similarity_type)

        # for each pairwise come, save value + label to pandas df
        similarity_data = DataFrame(np.column_stack((val, " ~ ".join([path[0], path[1]]))),
                                    columns=['similar_coef', 'image_labels'])
        coef_df = concat([coef_df, similarity_data], axis=0, ignore_index=True)

    return coef_df
