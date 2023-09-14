import numpy as np
import nibabel as nib
from pandas import DataFrame
from sklearn.preprocessing import minmax_scale
from pyrelimri.icc import sumsq_icc
from nilearn import image
from nilearn.maskers import (NiftiMasker,NiftiMapsMasker, NiftiLabelsMasker)
from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_destrieux_2009,
    fetch_atlas_difumo,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_msdl,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_smith_2009,
    fetch_atlas_talairach
)


def voxelwise_icc(multisession_list: str, mask: str, icc_type='icc_3'):
    """
    voxelwise_icc: calculates the ICC (+lower bound & upper bound CI)
    by voxel for specified input files using manual sumsq calculations.
    The path to the subject's data should be provided as a list of lists for sessions, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz",
        "./ses1/sub-03_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz",
        "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    dat_ses3 = ["./ses3/sub-00_Contrast-A_bold.nii.gz","./ses3/sub-01_Contrast-A_bold.nii.gz",
        "./ses3/sub-03_Contrast-A_bold.nii.gz"]

    ** The order of the subjects in each list has to be the same **

    Two session example:
    multisession_list  = [dat_ses1, dat_ses2]
    Three session example:
    multisession_list  = [dat_ses1, dat_ses2, dat_ses3]

    Inter-subject variance would be: between subjects in session 1, 2 & 3
    Intra-subject variance would be: within subject across session 1, 2 & 3.

    :param multisession_list: list of a list, a variable containing path to subjects 3D volumes for each session
    :param mask: path to nii MNI path object
    :param icc_type: provide icc type, default is icc_3, options: icc_1, icc_2, icc_3
    :return: returns 3D shaped array of ICCs in shape of provided 3D  mask
    """

    session_lengths = [len(session) for session in multisession_list]
    session_all_same = all(length == session_lengths[0] for length in session_lengths)

    assert session_all_same, f"Not all lists in session_files have the same length. " \
                             f"Mismatched lengths: {', '.join(str(length) for length in session_lengths)}"

    # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
    # iterates over list of lists
    try:
        session_data = [image.concat_imgs(i) for i in multisession_list]
    except ValueError as e:
        print(e)
        print("Error when attempting to concatenate images. Confirm affine/size of images.")

    # mask images
    masker = NiftiMasker(mask_img=mask)
    imgdata = [masker.fit_transform(i) for i in session_data]

    # get subj details per session to use in pandas df
    subj_n = imgdata[0].shape[:-1]
    subj_list = np.arange(subj_n[0])

    # calculate number of session, creating session labels and number of voxels
    num_sessions = len(imgdata)
    sess_labels = [f"sess{i + 1}" for i in range(num_sessions)]
    voxel_n = imgdata[0].shape[-1]

    # empty list for icc, low/upper bound 95% ICC, mean square between & within subject
    est, lowbound, upbound, msbs, msws = np.empty((5, voxel_n))

    for voxel in range(voxel_n):
        np_voxdata = np.column_stack((np.tile(subj_list, num_sessions),
                                      np.hstack(
                                          [[sess_labels[j]] * len(imgdata[j][:, voxel]) for j in range(num_sessions)]),
                                      np.hstack([imgdata[j][:, voxel] for j in range(num_sessions)])
                                      ))

        vox_pd = DataFrame(data=np_voxdata, columns=["subj", "sess", "vals"])
        vox_pd = vox_pd.astype({"subj": int, "sess": "category", "vals": float})

        est[voxel], lowbound[voxel], upbound[voxel], \
        msbs[voxel], msws[voxel] = sumsq_icc(df_long=vox_pd,
                                             sub_var="subj", sess_var="sess",
                                             value_var="vals", icc_type=icc_type)

    # using unmask to reshape the 1D voxels back to 3D specified mask and saving to dictionary
    result_dict = {
        'est': masker.inverse_transform(np.array(est)),
        'lower_bound': masker.inverse_transform(np.array(lowbound)),
        'upper_bound': masker.inverse_transform(np.array(upbound)),
        'ms_btwn': masker.inverse_transform(np.array(msbs)),
        'ms_wthn': masker.inverse_transform(np.array(msws))
    }

    return result_dict


def setup_atlas(name_atlas: str, **kwargs):
    default_params = {
        'data_dir': '/tmp/',
        'verbose': 0
    }

    # Dictionary mapping atlas names to their corresponding fetch functions
    grab_atlas = {
        'aal': fetch_atlas_aal,
        'destrieux_2009': fetch_atlas_destrieux_2009,
        'difumo': fetch_atlas_difumo,
        'harvard_oxford': fetch_atlas_harvard_oxford,
        'juelich': fetch_atlas_juelich,
        'msdl': fetch_atlas_msdl,
        'pauli_2017': fetch_atlas_pauli_2017,
        'shaefer_2018': fetch_atlas_schaefer_2018,
        'smith_2009': fetch_atlas_smith_2009,
        'talairach': fetch_atlas_talairach
    }
    try:
        atlas_grabbed = grab_atlas.get(name_atlas)
    except TypeError as e:
        print("Addition parameters required for atlas: {name_atlas}. Review: Nilearn Atlases for Details")
        print(e)

    if atlas_grabbed is None:
        raise ValueError(f"INCORRECT atlas name. PROVIDED: {name_atlas}\n"
                         f"OPTIONS: {', '.join(grab_atlas.keys())}")
    else:
        default_params.update(kwargs)
        return atlas_grabbed(**default_params)


def prob_atlas_scale(nifti_map, estimate_array):
    """
    Rescales a probabilistic 3D Nifti map to the range of estimated values.

    :param nifti_map: Nifti1Image (3D)
        The input Nifti image to be rescaled.
    :param estimate_array: ndarray (1D)
        A NumPy array containing the estimates used for scaling.
    :return: Nifti1Image
        Returns a 3D rescaled image based on the min/max of estimate_array.
    """

    temp_img_array = nifti_map.get_fdata().flatten()
    non_zero_mask = temp_img_array != 0

    # Scale the non-zero values using minmax_scale from sklearn
    scaled_values = minmax_scale(
        temp_img_array[non_zero_mask],
        feature_range=(min(estimate_array), max(estimate_array))
    )
    # New array w/ zeros & replace the non-zero values with the [new] scaled values
    rescaled = np.zeros_like(temp_img_array, dtype=float)
    rescaled[non_zero_mask] = scaled_values
    new_img_shape = np.reshape(rescaled, nifti_map.shape)

    return (image.new_img_like(nifti_map, new_img_shape))


def roi_icc(multisession_list: str, type_atlas: str,
            atlas_dir: str, icc_type='icc_3', **kwargs):
    """
    roi_icc: calculates the ICC for each ROI in atlas (+lower bound & upper bound CI)
        for specified input files using manual sumsq calculations.
    The path to the subject's data should be provided as a list of lists for sessions, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz",
        "./ses1/sub-03_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz",
        "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    dat_ses3 = ["./ses3/sub-00_Contrast-A_bold.nii.gz","./ses3/sub-01_Contrast-A_bold.nii.gz",
        "./ses3/sub-03_Contrast-A_bold.nii.gz"]

    ** The order of the subjects in each list has to be the same **

    Two session example:
    multisession_list  = [dat_ses1, dat_ses2]
    Three session example:
    multisession_list  = [dat_ses1, dat_ses2, dat_ses3]

    Inter-subject variance would be: between subjects in session 1, 2 & 3
    Intra-subject variance would be: within subject across session 1, 2 & 3.

    The atlas name is based on the probabilistic and ROI parcellations listed:
    https://nilearn.github.io/dev/modules/datasets.html#atlases

    :param multisession_list: list of a list, a variable containing path to subjects 3D volumes for each session
    :param type_atlas: name of atlas type provided within nilearn atlases
    :param atlas_dir: location to download/store downloaded atlas. Recommended: '/tmp/'
    :param icc_type: provide icc type, default is icc_3, options: icc_1, icc_2, icc_3
    :param **kwargs: each nilearn atlas has additional options, only defaults:
        data_dir = '/tmp', verbose = 0. These defaults will be updated with provided kwargs
    :return: returns 3D shaped array of ICCs in shape of provided 3D volumes
    """
    # grab atlas data
    try:
        atlas = setup_atlas(name_atlas=type_atlas, data_dir=atlas_dir, **kwargs)
    except TypeError as e:
        raise TypeError(f"Addition parameters required for atlas: {type_atlas}."
                        f"Review: Nilearn Atlases for Details. \nError: {e}")

    try:
        atlas_dim = len(atlas.maps.shape)
    except AttributeError:
        atlas_dim = len(nib.load(atlas.maps).shape)

    # combine brain data
    session_lengths = [len(session) for session in multisession_list]
    session_all_same = all(length == session_lengths[0] for length in session_lengths)

    assert session_all_same, f"Not all lists in session_files have the same length. " \
                             f"Mismatched lengths: {', '.join(str(length) for length in session_lengths)}"

    # concatenate the paths to 3D images into a 4D nifti image (4th dimension are subjs) using image concat
    try:
        session_data = [image.concat_imgs(i) for i in multisession_list]
    except ValueError as e:
        print(e)
        print("Error when attempting to concatenate images. Confirm affine/size of images.")

    # Atlases are either deterministic (3D) or probabilistic (4D). Try except to circumvent error
    # Get dimensions and then mask
    if atlas_dim == 3:
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=False,
            resampling_target='data',
            verbose=0
        ).fit()
    elif atlas_dim == 4:
        masker = NiftiMapsMasker(
            maps_img=atlas.maps,
            allow_overlap=True,
            standardize=False,
            resampling_target='data',
            verbose=0
        ).fit()
    else:
        raise ValueError("Atlas maps isn't 3D or 4D, so incompatible with Nifti[Labels/Maps]Masker() .")

    imgdata = [masker.transform(i) for i in session_data]

    # get subj details per session to use in pandas df
    subj_n = imgdata[0].shape[:-1]
    subj_list = np.arange(subj_n[0])

    # calculate number of session, creating session labels and number of voxels
    num_sessions = len(imgdata)
    sess_labels = [f"sess{i + 1}" for i in range(num_sessions)]
    roi_n = imgdata[0].shape[-1]

    # empty list for icc, low/upper bound 95% ICC, mean square between & within subject
    est, lowbound, upbound, msbs, msws = np.empty((5, roi_n))

    for roi in range(roi_n):
        np_roidata = np.column_stack((np.tile(subj_list, num_sessions),
                                      np.hstack(
                                          [[sess_labels[j]] * len(imgdata[j][:, roi]) for j in range(num_sessions)]),
                                      np.hstack([imgdata[j][:, roi] for j in range(num_sessions)])
                                      ))

        roi_pd = DataFrame(data=np_roidata, columns=["subj", "sess", "vals"])
        roi_pd = roi_pd.astype({"subj": int, "sess": "category", "vals": float})

        est[roi], lowbound[roi], upbound[roi], \
        msbs[roi], msws[roi] = sumsq_icc(df_long=roi_pd,
                                         sub_var="subj", sess_var="sess",
                                         value_var="vals", icc_type=icc_type)

    # using unmask to reshape the 1D ROI data back to 3D specified mask and saving to dictionary
    result_dict = {
        'roi_labels': atlas.labels,
        'est': np.array(est),
        'lower_bound': np.array(lowbound),
        'upper_bound': np.array(upbound),
        'ms_btwn': np.array(msbs),
        'ms_wthn': np.array(msws)
    }

    est_string = {"est_3d": est,
                  "lowbound_3d": lowbound, "upbound_3d": upbound,
                  "msbs_3d": msbs, "msws_3d": msws
                  }

    if atlas_dim == 4:
        for name, var in est_string.items():
            est_img = masker.inverse_transform(np.array(var))
            resample_img = prob_atlas_scale(est_img, np.array(var))
            result_dict[name] = resample_img
    else:
        update_values = {
            'est_3d': masker.inverse_transform(np.array(est)),
            'lower_bound_3d': masker.inverse_transform(np.array(lowbound)),
            'upper_bound_3d': masker.inverse_transform(np.array(upbound)),
            'ms_btwn_3d': masker.inverse_transform(np.array(msbs)),
            'ms_wthn_3d': masker.inverse_transform(np.array(msws))
        }
        result_dict.update(update_values)

    return result_dict


