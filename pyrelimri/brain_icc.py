import numpy as np
import nibabel as nib
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale
from pyrelimri.icc import sumsq_icc
from nilearn import image
from nilearn.maskers import (NiftiMasker, NiftiMapsMasker, NiftiLabelsMasker)
from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_destrieux_2009,
    fetch_atlas_difumo,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_msdl,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_talairach
)
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL = True


def process_voxel(voxel_data, subj_list, sess_labels, icc_type):
    """
    Process a single voxel's ICC calculation.
    
    Args:
        voxel_data: List of arrays, one per session containing voxel values for all subjects
        subj_list: Array of subject indices
        sess_labels: List of session labels
        icc_type: Type of ICC to compute
    
    Returns:
        Tuple of ICC metrics for this voxel
    """
    num_sessions = len(sess_labels)
    
    np_voxdata = np.column_stack((np.tile(subj_list, num_sessions),
                                  np.hstack([[sess_labels[j]] * len(voxel_data[j]) 
                                            for j in range(num_sessions)]),
                                  np.hstack(voxel_data)))

    vox_pd = DataFrame(data=np_voxdata, columns=["subj", "sess", "vals"])
    vox_pd = vox_pd.astype({"subj": int, "sess": "category", "vals": float})

    return sumsq_icc(df_long=vox_pd, sub_var="subj", sess_var="sess",
                     value_var="vals", icc_type=icc_type)


def voxelwise_icc(multisession_list: list, mask: str, icc_type: str = 'icc_3', n_jobs: int = -1) -> dict:
    """
    Calculate the Intraclass Correlation Coefficient (ICC) along with lower and upper bound confidence intervals
    by voxel for specified input files using manual sum of squares calculations. Now parallelized!

    Args:
        multisession_list (list of list of str):
            List of lists containing paths to subject 3D volumes for each session.

            Example:
                dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz", "./ses1/sub-01_Contrast-A_bold.nii.gz", "./ses1/sub-03_Contrast-A_bold.nii.gz"]
                dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz", "./ses2/sub-01_Contrast-A_bold.nii.gz", "./ses2/sub-03_Contrast-A_bold.nii.gz"]
                dat_ses3 = ["./ses3/sub-00_Contrast-A_bold.nii.gz", "./ses3/sub-01_Contrast-A_bold.nii.gz", "./ses3/sub-03_Contrast-A_bold.nii.gz"]
                The order of the subjects in each list has to be the same.

        mask (str):
            Path to 3D mask in NIfTI format.

        icc_type (str, optional):
            Type of ICC to compute, default is 'icc_3'.
            Options: 'icc_1', 'icc_2', 'icc_3'.

        n_jobs (int, optional):
            Number of parallel jobs. Default is -1 (use all available cores).

    Returns:
        dict:
            Dictionary containing the following 3D images:
                'est' (nibabel.Nifti1Image): Estimated ICC values.
                'lowbound' (nibabel.Nifti1Image): Lower bound of ICC confidence intervals.
                'upbound' (nibabel.Nifti1Image): Upper bound of ICC confidence intervals.
                'btwnsub' (nibabel.Nifti1Image): Between-subject variance.
                'wthnsub' (nibabel.Nifti1Image): Within-subject variance.
                'btwnmeas' (nibabel.Nifti1Image): Between-measurement variance.
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

    # Prepare data for parallel processing
    voxel_data_list = [[imgdata[j][:, voxel] for j in range(num_sessions)] 
                       for voxel in range(voxel_n)]

    # Parallel processing of voxels
    print(f"Processing {voxel_n} voxels using {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_voxel)(voxel_data, subj_list, sess_labels, icc_type)
        for voxel_data in voxel_data_list
    )

    # Unpack results
    results_array = np.array(results)
    est = results_array[:, 0]
    lowbound = results_array[:, 1] 
    upbound = results_array[:, 2]
    btwn_sub_var = results_array[:, 3]
    within_sub_var = results_array[:, 4]
    btwn_meas_var = results_array[:, 5]

    # using unmask to reshape the 1D voxels back to 3D specified mask and saving to dictionary
    result_dict = {
        'est': masker.inverse_transform(est),
        'lowbound': masker.inverse_transform(lowbound),
        'upbound': masker.inverse_transform(upbound),
        'btwnsub': masker.inverse_transform(btwn_sub_var),
        'wthnsub': masker.inverse_transform(within_sub_var),
        'btwnmeas': masker.inverse_transform(btwn_meas_var)
    }

    return result_dict


def setup_atlas(name_atlas: str, **kwargs) -> nib.Nifti1Image:
    """
    Setup & fetch a brain atlas based on the provided atlas name & optional parameters via kwargs associated
    with documentation from Nilearn.

    Args:
        name_atlas (str):
            Name of the atlas to fetch. Available options are:
            'aal', 'destrieux_2009', 'difumo', 'harvard_oxford', 'juelich',
            'msdl', 'pauli_2017', 'schaefer_2018', 'talairach'.

        **kwargs:
            Additional parameters to customize the fetching process. Examples:
                - 'data_dir' (str): Directory where the fetched atlas data will be stored. Default is '/tmp/'.
                - 'verbose' (int): Verbosity level of the process. Default is 0.

    Returns:
        nib.Nifti1Image:
            Fetched brain atlas in NIfTI format.
    """
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
    Rescales a probabilistic 3D Nifti map to match the range of estimated values.

    Args:
        nifti_map (Nifti1Image):
            Input 3D Nifti image to be rescaled.

        estimate_array (ndarray):
            1D NumPy array containing the estimates used for scaling.

    Returns:
        Nifti1Image:
            Rescaled 3D image where non-zero values are scaled to match the range of `estimate_array`.

    Notes:
        This function rescales the non-zero values in the input Nifti image `nifti_map` using the minimum and maximum
        values of `estimate_array`. The spatial/header info from `nifti_map` is preserved.
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

    return image.new_img_like(nifti_map, new_img_shape)


def roi_icc(multisession_list: list, type_atlas: str, atlas_dir: str, icc_type='icc_3', **kwargs):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for each ROI in a specified atlas
    (+lower bound & upper bound CI) for input files using manual sum of squares calculations.
    It also provides associated between subject variance, within subject variance and between
    measure variance estimates.
    The function expects the subject's data paths to be provided as a list of lists for sessions:

    Example:
        dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz", "./ses1/sub-01_Contrast-A_bold.nii.gz", "./ses1/sub-03_Contrast-A_bold.nii.gz"]
        dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz", "./ses2/sub-01_Contrast-A_bold.nii.gz", "./ses2/sub-03_Contrast-A_bold.nii.gz"]
        dat_ses3 = ["./ses3/sub-00_Contrast-A_bold.nii.gz", "./ses3/sub-01_Contrast-A_bold.nii.gz", "./ses3/sub-03_Contrast-A_bold.nii.gz"]
        ** The order of the subjects in each list has to be the same **

    Examples:
        # Two-session example:
        multisession_list = [dat_ses1, dat_ses2]
        # Three-session example:
        multisession_list = [dat_ses1, dat_ses2, dat_ses3]

    Inter-subject variance corresponds to variance between subjects across all sessions (1, 2, 3).
    Intra-subject variance corresponds to variance within subjects across all sessions (1, 2, 3).

    The atlas name should be one of the probabilistic and ROI parcellations listed:
    https://nilearn.github.io/dev/modules/datasets.html#atlases

    Args:
        multisession_list (list of list of str): List of lists containing paths to subject 3D volumes for each session.
        type_atlas (str): Name of the atlas type provided within Nilearn atlases.
        atlas_dir (str): Location to download/store downloaded atlas. Recommended: '/tmp/'.
        icc_type (str, optional): Type of ICC to compute, default is 'icc_3'. Options: 'icc_1', 'icc_2', 'icc_3'.
        **kwargs (optional): Additional parameters to customize the atlas fetching process and masker
            settings.
            - data_dir (str): Directory where the fetched atlas data will be stored. Default is '/tmp/'.
            - verbose (int): Verbosity level of the fetching process. Default is 0.

    Returns:
        dict: Dictionary containing the following arrays and values:
            - roi_labels (list): Labels of the ROIs in the atlas.
            - est (ndarray): Estimated ICC values for each ROI.
            - lowbound (ndarray): Lower bound of ICC confidence intervals for each ROI.
            - upbound (ndarray): Upper bound of ICC confidence intervals for each ROI.
            - btwnsub (ndarray): Between-subject variance for each ROI.
            - wthnsub (ndarray): Within-subject variance for each ROI.
            - btwnmeas (ndarray): Between-measurement variance for each ROI.
            - est_3d (nibabel.Nifti1Image): Estimated ICC values for each ROI.
            - lowbound_3d (nibabel.Nifti1Image): Lower bound of ICC confidence intervals for each ROI.
            - upbound_3d (nibabel.Nifti1Image): Upper bound of ICC confidence intervals for each ROI.
            - btwnsub_3d (nibabel.Nifti1Image): Between-subject variance for each ROI.
            - wthnsub_3d (nibabel.Nifti1Image): Within-subject variance for each ROI.
            - btwnmeas_3d (nibabel.Nifti1Image): Between-measurement variance for each ROI.

    Example:
        # Calculate ICC for ROIs using multisession data and AAL atlas
        result = roi_icc(multisession_list=multisession_list, type_atlas='aal', atlas_dir='/tmp/', icc_type='icc_2')
    """
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

    # grab atlas data
    try:
        atlas = setup_atlas(name_atlas=type_atlas, data_dir=atlas_dir, **kwargs)
    except TypeError as e:
        raise TypeError(f"Addition parameters required for atlas: {type_atlas}."
                        f"Review: Nilearn Atlases for Details. \nError: {e}")

    # Atlases are either deterministic (3D) or probabilistic (4D). Try except to circumvent error
    # Get dimensions and then mask
    try:
        atlas_dim = len(atlas.maps.shape)
    except AttributeError:
        atlas_dim = len(nib.load(atlas.maps).shape)

    if atlas_dim == 3:
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=False,
            resampling_target='labels',
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

    # empty list for icc, low/upper bound 95% ICC, between sub, within sub and between measure var
    est, lowbound, upbound, \
        btwn_sub_var, within_sub_var, btwn_meas_var = np.empty((6, roi_n))

    for roi in range(roi_n):
        np_roidata = np.column_stack((np.tile(subj_list, num_sessions),
                                      np.hstack(
                                          [[sess_labels[j]] * len(imgdata[j][:, roi]) for j in range(num_sessions)]),
                                      np.hstack([imgdata[j][:, roi] for j in range(num_sessions)])
                                      ))

        roi_pd = DataFrame(data=np_roidata, columns=["subj", "sess", "vals"])
        roi_pd = roi_pd.astype({"subj": int, "sess": "category", "vals": float})

        est[roi], lowbound[roi], upbound[roi], \
            btwn_sub_var[roi], within_sub_var[roi], \
            btwn_meas_var[roi] = sumsq_icc(df_long=roi_pd, sub_var="subj", sess_var="sess",
                                           value_var="vals", icc_type=icc_type)

    # using unmask to reshape the 1D ROI data back to 3D specified mask and saving to dictionary
    result_dict = {
        'roi_labels': atlas.labels[1:],
        'est': np.array(est),
        'lowbound': np.array(lowbound),
        'upbound': np.array(upbound),
        'btwnsub': np.array(btwn_sub_var),
        'wthnsub': np.array(within_sub_var),
        'btwnmeas': np.array(btwn_meas_var)
    }

    est_string = {"est_3d": est,
                  "lowbound_3d": lowbound, "upbound_3d": upbound,
                  "btwnsub_3d": btwn_sub_var, "wthnsub_3d": within_sub_var,
                  "btwnmeas_3d": btwn_meas_var
                  }

    if atlas_dim == 4:
        for name, var in est_string.items():
            est_img = masker.inverse_transform(np.array(var))
            resample_img = prob_atlas_scale(est_img, np.array(var))
            result_dict[name] = resample_img
    else:
        update_values = {
            'est_3d': masker.inverse_transform(np.array(est)),
            'lowbound_3d': masker.inverse_transform(np.array(lowbound)),
            'upbound_3d': masker.inverse_transform(np.array(upbound)),
            'btwnsub_3d': masker.inverse_transform(np.array(btwn_sub_var)),
            'wthnsub_3d': masker.inverse_transform(np.array(within_sub_var)),
            'btwnmeas_3d': masker.inverse_transform(np.array(within_sub_var))
        }
        result_dict.update(update_values)

    return result_dict
