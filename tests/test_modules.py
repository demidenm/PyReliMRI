import pytest
import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import seaborn as sns
from pyrelimri.similarity import image_similarity
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from pyrelimri.brain_icc import (voxelwise_icc, setup_atlas, roi_icc)
from pyrelimri.icc import sumsq_icc
from pyrelimri.masked_timeseries import (trlocked_events, extract_time_series_values,
                                         extract_time_series, extract_postcue_trs_for_conditions)
from collections import namedtuple
from nilearn.datasets import fetch_neurovault_ids
from nilearn.masking import compute_multi_brain_mask
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import warnings
warnings.filterwarnings("ignore", message="The nilearn.glm module is experimental.", category=FutureWarning)
from nilearn.glm.first_level import make_first_level_design_matrix


def generate_img_pair(r: float, dir: Path, use_mask: bool = False, tol: float = .001,
                      imgdims: list = None,
                      maskdims: list = None) -> namedtuple:
    """
    r: correlation bw images
    dir: Path for saving files
    use_mask: bool, create mask and mask data
    tol: tolerance for correlation value - lower than .001 will make it really slow
    returns:
        images: path to two image files with specified correlation
        mask: path to mask image, all ones if use_mask==False
    """
    imgpair = namedtuple("ImgPair", "tol r images maskimg")
    imgpair.images = []
    imgpair.r = r
    imgpair.tol = tol

    rng = np.random.default_rng()

    if imgdims is None:
        imgdims = [64, 64, 32]
    nvox = np.prod(imgdims)
    if use_mask:
        if maskdims is None:
            maskdims = [round(i/2) for i in imgdims]
        mask = np.zeros(imgdims).astype('int')
        mask[:maskdims[0], :maskdims[1], :maskdims[2]] = 1
    else:
        mask = np.ones(imgdims)
        
    maskvox = mask.reshape(nvox)

    empirical_r = 10
    while (np.abs(empirical_r - r) > tol):
        data = rng.multivariate_normal(mean=[0, 0],
                                       cov=[[1, r], [r, 1]], size=nvox)
        empirical_r = np.corrcoef(data[maskvox == 1, :].T)[0, 1]
    
    for i in range(2):
        imgpair.images.append(dir / f'testimg_{i}.nii.gz')
        tmpimg = nib.Nifti1Image((maskvox * data[:, i]).reshape(imgdims),
                                 affine=np.eye(4))
        tmpimg.to_filename(imgpair.images[-1])
    imgpair.mask = dir / 'mask.nii.gz'
    maskimg = nib.Nifti1Image(mask, affine=np.eye(4))
    maskimg.to_filename(imgpair.mask)
    return imgpair
    

@pytest.fixture(scope="session")
def image_pair(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")
    return generate_img_pair(r=0.5, dir=tmpdir)


def test_image_pair_smoke(image_pair):
    assert image_pair.images is not None


def test_image_pair_images(image_pair):
    for imgfile in image_pair.images + [image_pair.mask]:
        img = nib.load(imgfile)
        assert img is not None

def create_dummy_nifti(shape, affine, filepath):
    data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(filepath))


def test_session_lengths_mismatch(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    # Test case with different session lengths
    multisession_list = [
        [tmpdir / "sub-00_ses1_Contrast-A_bold.nii.gz",
         tmpdir / "sub-01_ses1_Contrast-A_bold.nii.gz"],
        [tmpdir / "sub-00_ses2_Contrast-A_bold.nii.gz",
         tmpdir / "sub-01_ses2_Contrast-A_bold.nii.gz",
         tmpdir / "sub-03_ses2_Contrast-A_bold.nii.gz"]
    ]

    icc_type = "icc_3"

    # Create dummy NIfTI files
    shape = (97, 115, 97)
    affine = np.eye(4)
    for session in multisession_list:
        for filepath in session:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            create_dummy_nifti(shape, affine, filepath)

    mask = compute_multi_brain_mask(target_imgs=[
        tmpdir / "sub-00_ses2_Contrast-A_bold.nii.gz",
        tmpdir / "sub-01_ses2_Contrast-A_bold.nii.gz",
        tmpdir / "sub-03_ses2_Contrast-A_bold.nii.gz"
    ])

    mask_path = tmpdir / 'test_mask.nii.gz'
    nib.save(mask, mask_path)

    # The assertion should raise an exception
    with pytest.raises(AssertionError):
        voxelwise_icc(multisession_list, mask, icc_type)


@pytest.mark.parametrize("measure", ['Dice', 'Jaccard'])
def test_image_similarity(image_pair, measure):
    imgsim = image_similarity(
        image_pair.images[0], image_pair.images[1], image_pair.mask,
        thresh=1, similarity_type=measure
    )
    assert imgsim is not None

def test_spearman_similarity(image_pair):
    spearman_sim = image_similarity(
        image_pair.images[0], image_pair.images[1], similarity_type='spearman')
    assert abs(spearman_sim - 0.5) < 0.1, "The similarity is not close to 0.5."

def test_image_similarity_spearman_value_error(image_pair):
    imgfile1 = image_pair.images[0]
    imgfile2 = image_pair.images[1]
    mask = None
    thresh = 0.5
    similarity_type = "spearman"
    with pytest.raises(ValueError):
        image_similarity(imgfile1, imgfile2, mask=mask, thresh=thresh, similarity_type=similarity_type)

#@pytest.mark.parametrize("corr", [.50, .60,.45])
def test_calculate_icc1():
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = sumsq_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess", value_var="vals", icc_type='icc_1')

    assert np.allclose(icc[0], -0.05, atol=.01)


def test_calculate_icc2():
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = sumsq_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess",value_var="vals", icc_type='icc_2')
    assert np.allclose(icc[0], 0.11, atol=.01)

# tetrachoric tests
def test_tetrachoric_corr():
    assert np.allclose(
        tetrachoric_corr(np.array([0, 0, 1, 1]),
                         np.array([0, 1, 0, 1])),
        0.0)
    assert np.allclose(
        tetrachoric_corr(np.array([0, 0, 1, 1]),
                         np.array([0, 0, 1, 1])),
        1.0)
    assert np.allclose(
        tetrachoric_corr(np.array([0, 0, 1, 1]),
                         np.array([1, 1, 0, 0])),
        -1.0)

def test_tetrachoric_corr_nanhandling():
    assert np.isnan(
        tetrachoric_corr(np.array([0, 0, 1, 1]),
                         np.array([1, 1, 1, 1])))

# property based testing with a range of arrays
@given(vec=arrays(np.int8, (2, 24), elements=st.integers(0, 100)))
def test_tetrachoric_corr_hypothesis(vec):
    tc = tetrachoric_corr(vec[0, :], vec[1, :])
    if (vec[0, :] == vec[1, :]).all():
        assert tc == 1.0
    else:
        B = sum(np.logical_and(vec[0, :] == 0, vec[1, :] == 1))
        C = sum(np.logical_and(vec[0, :] == 1, vec[1, :] == 0))
        # should return nan in these cases
        if B == 0 or C == 0:
            assert np.isnan(tc)
        else:
            assert tc <= 1.0 and tc >= -1.0

# test roi based ICC
def setup_atlas_valuerrror():
    with pytest.raises(ValueError):
        setup_atlas(name_atlas='fake_atlas')

@pytest.mark.parametrize("atlases", ['aal', 'difumo'])
def setup_atlas_noerror(atlases):
    setup_atlas(name_atlas=atlases)

def test_roiicc_msc(tmp_path_factory):

    # create temp dir
    tmpdir = tmp_path_factory.mktemp("data")

    # Test case w/ neurovault data
    MSC01_1 = fetch_neurovault_ids(image_ids=[48068], data_dir=tmpdir)  # MSC01 motor session1 1 L Hand beta
    MSC01_2 = fetch_neurovault_ids(image_ids=[48073], data_dir=tmpdir)  # MSC01 motor session2 1 L Hand beta
    MSC02_1 = fetch_neurovault_ids(image_ids=[48118], data_dir=tmpdir)
    MSC02_2 = fetch_neurovault_ids(image_ids=[48123], data_dir=tmpdir)
    MSC03_1 = fetch_neurovault_ids(image_ids=[48168], data_dir=tmpdir)
    MSC03_2 = fetch_neurovault_ids(image_ids=[48173], data_dir=tmpdir)

    ses1 = [MSC01_1['images'], MSC02_1['images'], MSC03_1['images']]
    ses2 = [MSC01_2['images'], MSC02_2['images'], MSC03_2['images']]

    # estimate ICC for roi = 200 in shaefer
    result = roi_icc(multisession_list=[ses1, ses2], type_atlas='shaefer_2018',
                     atlas_dir = tmpdir, icc_type='icc_3')

    assert np.allclose(result['est'][200], .70, atol=.01)


# tests for masked timeseries
def test_miss_sub_boldpath():
    bold_paths = ['/tmp/NDA_run-01_bold.nii.gz']
    roi_type = 'mask'
    roi_mask = '/tmp/roi_mask.nii.gz'

    with pytest.raises(AssertionError):
        extract_time_series(bold_paths, roi_type, roi_mask=roi_mask)

def test_miss_sub_boldpath():
    bold_paths = ['/tmp/sub-NDA_01_bold.nii.gz']
    roi_type = 'mask'
    roi_mask = '/tmp/roi_mask.nii.gz'

    with pytest.raises(AssertionError):
        extract_time_series(bold_paths, roi_type, roi_mask=roi_mask)


def test_mismatched_shapes_boldroi(tmp_path):
    # testing that error is thrown when mask and BOLD images are not similar shape
    bold_path = tmp_path / "sub-01_run-01_bold.nii.gz"
    roi_mask_path = tmp_path / "roi_mask.nii.gz"

    # Create dummy BOLD and ROI NIfTI files with mismatched shapes
    create_dummy_nifti((64, 64, 36, 2), np.eye(4), bold_path)
    create_dummy_nifti((64, 64, 34), np.eye(4), roi_mask_path)

    bold_paths = [bold_path]
    roi_type = 'mask'

    # Ensure that AssertionError is raised when shapes are mismatched
    with pytest.raises(AssertionError):
        extract_time_series(bold_paths, roi_type, roi_mask=str(roi_mask_path))

def test_wrongorder_behbold_ids():
    # testing that order of sub & run paths for BOLD paths != Behavioral paths
    bold_path_list = [f"sub-0{i:02d}_run-01.nii.gz" for i in range(20)] + \
                     [f"sub-0{i:02d}_run-02.nii.gz" for i in range(20)]
    beh_path_list = [f"sub-0{i:02d}_run-01.nii.gz" for i in range(15)] + \
                     [f"sub-0{i:02d}_run-02.nii.gz" for i in range(20)]

    with pytest.raises(AssertionError):
        extract_postcue_trs_for_conditions(events_data=beh_path_list, onset='Test',
                                           trial_name='test', bold_tr=.800, bold_vols=200,
                                           time_series=[0,1,2,3], conditions=['test'], tr_delay=15,
                                           list_trpaths= bold_path_list)

def test_wrongroi_type():
    # Define invalid ROI type
    wrong_roi_lab = 'fookwrng'

    # Define other function arguments
    bold_paths = ["sub-01_run-01_bold.nii.gz"]
    high_pass_sec = 100
    roi_mask = "roi_mask.nii.gz"

    # Test if ValueError is raised for invalid ROI type
    with pytest.raises(ValueError):
        extract_time_series(bold_paths, wrong_roi_lab, high_pass_sec=high_pass_sec, roi_mask=roi_mask)

def test_missing_file():
    # test when events file is not found
    events_path = "missing_file_name.csv"
    onsets_column = "onsets"
    trial_name = "trial"
    bold_tr = 2.0
    bold_vols = 10

    # Test if FileNotFoundError is raised when the file does not exist
    with pytest.raises(FileNotFoundError):
        trlocked_events(events_path, onsets_column, trial_name, bold_tr, bold_vols)


def test_missing_eventscol(tmp_path):
    # testing missing column "trial" in events file
    events_path = tmp_path / "test_events.csv"
    with open(events_path, "w") as f:
        f.write("onsets\n0.0\n1.0\n2.0\n")

    # Define function arguments
    onsets_column = "onsets"
    trial_name = "trial"
    bold_tr = 2.0
    bold_vols = 10

    # Test if KeyError is raised when columns are missing
    with pytest.raises(KeyError):
        trlocked_events(events_path, onsets_column, trial_name, bold_tr, bold_vols)


def test_lenbold_mismatchtrlen(tmp_path):
    # The length of the resulting TR locked values (length) should be similar N to BOLD.
    # assume to always be true but confirm
    events_path = tmp_path / "test_events.csv"
    onset_name = 'onsets'
    trial_name = 'trial'
    bold_tr = 2.0
    bold_vols = 5  # Mismatched bold_vols compared to the expected length of merged_df

    with open(events_path, "w") as f:
        f.write(f"{onset_name},{trial_name}\n0.0,Test1\n1.0,Test2\n2.0,Test1\n")

    with pytest.raises(ValueError):
        trlocked_events(events_path=events_path, onsets_column=onset_name, trial_name=trial_name,
                        bold_tr=bold_tr, bold_vols=bold_vols, separator=',')

def test_runtrlocked_events(tmp_path):
    # The length of the resulting TR locked values (length) should be similar N to BOLD.
    # assume to always be true but confirm
    events_path = tmp_path / "test_events.csv"
    onset_name = 'onsets'
    trial_name = 'trial'
    bold_tr = 2.0
    bold_vols = 3
    with open(events_path, "w") as f:
        f.write(f"{onset_name},{trial_name}\n0.0,Test1\n2.0,Test2\n4.0,Test1\n")

    trlocked_events(events_path=events_path, onsets_column=onset_name, trial_name=trial_name,
                    bold_tr=bold_tr, bold_vols=bold_vols, separator=',')


def create_conv_mat(eventsdf, tr_dur=None, acq_dur=None):
    vol_time = acq_dur
    tr = tr_dur
    design_mat = make_first_level_design_matrix(
        frame_times=np.linspace(0, vol_time, int(vol_time/tr)),
        events=eventsdf, hrf_model='spm',
        drift_model=None, high_pass=None)
    return design_mat


@pytest.mark.parametrize("TR", [.8, 1.4, 2, 2.6])
@pytest.mark.parametrize("interval", [10, 15, 20])
def test_testsimtrpeak(tmp_path, TR, interval):
    onsets = np.arange(0, 160, interval)
    dur_opts = [1.5, 2, 2.5]
    prob_durs = [.50, .25, .25]
    np.random.seed(11)
    durations = np.random.choice(
        dur_opts, size=len(onsets), p=prob_durs
    )

    events_df = pd.DataFrame({
        "onset": onsets,
        "duration": durations,
        "trial_type": "Phacking101"
    })
    last_onset = events_df['onset'].iloc[-1]
    tr = TR
    conv_vals = create_conv_mat(eventsdf=events_df, tr_dur=tr, acq_dur=last_onset)

    # create n = 1 compatible timeseries for test
    convolved_stacked = np.vstack([conv_vals['Phacking101']])
    convolved_stacked = convolved_stacked.reshape((conv_vals.shape[0] * (conv_vals.shape[1] - 1), 1))
    timeseries_reshaped = np.reshape(convolved_stacked, (1, len(convolved_stacked), 1))

    events_file_name = tmp_path / "sub-01_run-01_test-events.csv"
    events_df.to_csv(events_file_name, sep='\t')

    conditions = ['Phacking101']
    trdelay = int(15 / tr)
    df = extract_postcue_trs_for_conditions(events_data=[events_file_name], onset='onset', trial_name='trial_type',
                                            bold_tr=TR, bold_vols=len(timeseries_reshaped[0]),
                                            time_series=timeseries_reshaped,
                                            conditions=conditions, tr_delay=trdelay,
                                            list_trpaths=['sub-01_run-01'])
    df['Mean_Signal'] = pd.to_numeric(df['Mean_Signal'], errors='coerce') # to avoid argmax object error
    tr_peak = df.loc[df['Mean_Signal'].idxmax(), 'TR']
    min_tr = np.floor(float(6 / tr))
    max_tr = np.ceil(float(10 / tr))
    peak_in_tr = np.arange(min_tr, max_tr, .1)
    is_in_array = np.any(np.isclose(peak_in_tr, tr_peak))
    print(f"Checking whether {tr_peak} TR HRF peak is between range min {min_tr} and max {max_tr}")
    assert is_in_array, f"Peak error: Peak should occurs between 5-8sec, peak {round(tr_peak * tr, 2)}"