import numpy as np
from pandas import DataFrame
from pyrelimri.icc import sumsq_icc
from nilearn import image
from nilearn.maskers import NiftiMasker


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
    session_data = [image.concat_imgs(i) for i in multisession_list]

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
