import numpy as np
from pandas import DataFrame
from pyrelimri.icc import sumsq_icc
from nilearn import image
from nilearn.maskers import NiftiMasker


def voxelwise_icc(paths_sess1: str, paths_sess2: str, mask: str, paths_sess3=None, icc_type='icc_3'):
    """
    voxelwise_icc: calculates the ICC (+lower bound & upper bound CI)
    by voxel for specified input files using manual sumsq calculations.
    The path to the subject's data should be provided as a list for each session, i.e.
    dat_ses1 = ["./ses1/sub-00_Contrast-A_bold.nii.gz","./ses1/sub-01_Contrast-A_bold.nii.gz",
        "./ses1/sub-03_Contrast-A_bold.nii.gz"]
    dat_ses2 = ["./ses2/sub-00_Contrast-A_bold.nii.gz","./ses2/sub-01_Contrast-A_bold.nii.gz",
        "./ses2/sub-03_Contrast-A_bold.nii.gz"]
    Inter-subject variance would be: between subjects in session 1 & between subjects in session 2
    Intra-subject variance would be: within subject across session 1 and session 2.

    :param paths_sess1: paths to session 2 nii MNI files
    :param paths_sess2: paths to session 2 nii MNI files
    :param paths_sess3: If there are more than 3 sessions, paths to session 3 nii MNI files
    :param mask: path to nii MNI path object
    :param icc_type: provide icc type, default is icc_3, options: icc_1, icc_2, icc_3
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

    # empty list for icc, low/upper bound 95% ICC, mean square between & within subject
    icc_calc = []
    icc_lb = []
    icc_ub = []
    msbs = []
    msws = []

    if paths_sess3 is None:
        for voxel in range(len(imgdata[0].T)):
            # sub sample v voxel for all subjects for each session. Stack columns to create np array
            #  that includes voxels and sub labels (1-n, repeated) for session 1 [0] & session 2 [1]
            np_voxdata = np.column_stack((np.tile(sub_n, 2),
                                          np.hstack((["sess1"] * len(imgdata[0][:, voxel]),
                                                     ["sess2"] * len(imgdata[1][:, voxel]))),
                                          np.hstack((imgdata[0][:, voxel], imgdata[1][:, voxel]))
                                          ))

            # create dataframe that is then used with ICC function to calculate specified ICC
            vox_pd = DataFrame(data=np_voxdata, columns=["subj", "sess", "vals"])
            vox_pd = vox_pd.astype({"subj": int, "sess": "category", "vals": float})

            iccest, icclb, iccub, MSbtw, MSwtn = sumsq_icc(df_long=vox_pd, sub_var="subj", sess_var="sess",
                                                           value_var="vals", icc_type=icc_type)
            icc_calc.append(iccest)
            icc_lb.append(icclb)
            icc_ub.append(iccub)
            msbs.append(MSbtw)
            msws.append(MSwtn)

    elif paths_sess3 is not None:
        for voxel in range(len(imgdata[0].T)):
            # sub sample v voxel for all subjects for each session. Stack columns to create np array
            #  that includes voxels and sub labels for session 1 [0] & session 2 [1], session [3]
            np_voxdata = np.column_stack((np.tile(sub_n, 3),
                                          np.hstack((["sess1"] * len(imgdata[0][:, voxel]),
                                                     ["sess2"] * len(imgdata[1][:, voxel]),
                                                     ["sess3"] * len(imgdata[2][:, voxel]))),
                                          np.hstack((imgdata[0][:, voxel], imgdata[1][:, voxel]))
                                          ))

            # create dataframe that is then used with ICC function to calculate specified ICC
            vox_pd = DataFrame(data=np_voxdata, columns=["subj", "sess", "vals"])
            vox_pd = vox_pd.astype({"subj": int, "sess": "category", "vals": float})

            iccest, icclb, iccub, MSbtw, MSwtn = sumsq_icc(df_long=vox_pd, sub_var="subj", sess_var="sess",
                                                           value_var="vals", icc_type=icc_type)
            icc_calc.append(iccest)
            icc_lb.append(icclb)
            icc_ub.append(iccub)
            msbs.append(MSbtw)
            msws.append(MSwtn)

        # using unmask to reshape the 1D voxels back to 3D specified mask

    return masker.inverse_transform(np.array(icc_calc)), masker.inverse_transform(np.array(icc_lb)), \
           masker.inverse_transform(np.array(icc_ub)), masker.inverse_transform(np.array(msbs)), \
           masker.inverse_transform(np.array(msws))
