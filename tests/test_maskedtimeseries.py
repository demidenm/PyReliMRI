import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pyrelimri.masked_timeseries import (trlocked_events, extract_time_series, extract_postcue_trs_for_conditions)
import warnings
warnings.filterwarnings("ignore", message="The nilearn.glm module is experimental.", category=FutureWarning)
from nilearn.glm.first_level import make_first_level_design_matrix


def create_dummy_nifti(shape, affine, filepath):
    data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(filepath))


def test_miss_sub_boldpath():
    bold_paths = ['/tmp/NDA_run-01_bold.nii.gz']
    roi_type = 'mask'
    roi_mask = '/tmp/roi_mask.nii.gz'

    with pytest.raises(ValueError):
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
                                           time_series=[0, 1, 2, 3], conditions=['test'], tr_delay=15,
                                           list_trpaths=bold_path_list)

def test_wrongroi_type():
    # Define invalid ROI type
    wrong_roi_lab = 'Testinit'

    # Define other function arguments
    bold_paths = ["sub-01_run-01_bold.nii.gz"]
    high_pass_sec = 100
    roi_mask = "roi_mask.nii.gz"

    # Test if ValueError is raised for invalid ROI type
    with pytest.raises(ValueError):
        extract_time_series(bold_paths, wrong_roi_lab, high_pass_sec=high_pass_sec, roi_mask=roi_mask)

def test_missing_file():
    # test when events file is not found
    events_path = "missing_file_testin-it.csv"
    onsets_column = "onsets"
    trial_name = "trial"
    bold_tr = 2.0
    bold_vols = 10

    # Test if FileNotFoundError is raised when the file does not exist
    with pytest.raises(FileNotFoundError):
        trlocked_events(events_path, onsets_column, trial_name, bold_tr, bold_vols)


def test_missing_eventscol(tmp_path):
    # testing missing column "trial" in events file
    events_path = tmp_path / "testin-it_events.csv"
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
    events_path = tmp_path / "testin-it_events.csv"
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
    events_path = tmp_path / "testin-it_events.csv"
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
        "trial_type": "Testinit"
    })
    last_onset = events_df['onset'].iloc[-1]
    tr = TR
    conv_vals = create_conv_mat(eventsdf=events_df, tr_dur=tr, acq_dur=last_onset)

    # create n = 1 compatible timeseries for test
    convolved_stacked = np.vstack([conv_vals['Testinit']])
    convolved_stacked = convolved_stacked.reshape((conv_vals.shape[0] * (conv_vals.shape[1] - 1), 1))
    timeseries_reshaped = np.reshape(convolved_stacked, (1, len(convolved_stacked), 1))

    events_file_name = tmp_path / "sub-01_run-01_test-events.csv"
    events_df.to_csv(events_file_name, sep='\t')

    conditions = ['Testinit']
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