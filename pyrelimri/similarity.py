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
    Calculate the similarity between two 3D images using a specified similarity metric.
    The function computes the ratio of intersecting and union voxels based on the provided threshold and similarity type
    The result is a similarity coefficient indicating the overlap between the two images.

    Parameters
    ----------
    imgfile1 : str
        Path to the first NIfTI image file.

    imgfile2 : str
        Path to the second NIfTI image file.

    mask : str, optional
        Path to a binarized mask image for voxel selection. Default is None.

    thresh : float, optional
        Threshold value for voxel selection. Positive values retain voxels greater than the threshold,
        and negative values retain voxels less than the threshold. Default is None.

    similarity_type : str, optional
        Similarity calculation method. Options are 'dice', 'jaccard', 'tetrachoric', or 'spearman'. Default is 'dice'.

    Returns
    -------
    float
        Similarity coefficient based on the selected method.


    Example
    -------
    # Example usage of image_similarity
    similarity = image_similarity(imgfile1='./img1.nii', imgfile2='./img2.nii',
    mask='./mask.nii', thresh=0.5, similarity_type='dice')
    """
    assert similarity_type.casefold() in ['dice', 'jaccard',
                                          'tetrachoric', 'spearman'], 'similarity_type must be ' \
                                                                      '"Dice", "Jaccard", "Tetrachoric" or ' \
                                                                      '"Spearman". Provided: {}"'.format(similarity_type)

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
        
        if similarity_type.casefold() == 'dice':
            # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
            sum_a_b = imgdata[0, :].sum() + imgdata[1, :].sum()
            coeff = (2.0 * intersect.sum()) / (float(sum_a_b) + np.finfo(float).eps)
        else:
            # Jaccard coefficient: |A ∩ B| / |A ∪ B|
            union = np.logical_or(imgdata[0, :], imgdata[1, :])
            coeff = intersect.sum() / (float(union.sum()) + np.finfo(float).eps)
            
    elif similarity_type.casefold() == 'tetrachoric':
        warnings.filterwarnings('ignore')
        coeff = tet_corr(vec1=imgdata[0, :], vec2=imgdata[1, :])

    else:
        if thresh is not None:
            raise ValueError(f"Spearman rank should be for unthresholded images."
                             f"/n Threshold is set to: {thresh}./n Advise: 'None'.")
        else:
            coeff = spearmanr(a=imgdata[0, :], b=imgdata[1, :])[0]

    return coeff


def pairwise_similarity(nii_filelist: list, mask: str = None,
                        thresh: float = None, similarity_type: str = 'Dice') -> DataFrame:
    """
    Calculate pairwise similarity between a list of NIfTI images using a specified similarity metric.
    The function generates all possible combinations of the provided NIfTI images and computes the similarity
    coefficient for each pair.

    Parameters
    ----------
    nii_filelist : list
        List of paths to NIfTI image files.

    mask : str, optional
        Path to the brain mask image for voxel selection. Default is None.

    thresh : float, optional
        Threshold value for voxel selection. Positive values retain voxels greater than the threshold,
        and negative values retain voxels less than the threshold. Default is None.

    similarity_type : str, optional
        Similarity calculation method. Options are 'dice', 'jaccard', 'tetrachoric', or 'spearman'. Default is 'dice'.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the similarity coefficients and corresponding image labels for each pairwise comparison.


    Example
    -------
    # Example usage of pairwise_similarity
    similarity_df = pairwise_similarity(['./img1.nii', './img2.nii', './img3.nii'],
    mask='mask.nii', thresh=0.5, similarity_type='dice')
    """
    # test whether function type is of 'Dice' or 'Jaccard', case insensitive
    assert similarity_type.casefold() in ['dice', 'jaccard',
                                          'tetrachoric', 'spearman'], 'similarity_type must be ' \
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
