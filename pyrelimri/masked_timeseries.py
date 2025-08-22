import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nibabel import Nifti1Image
from nilearn.maskers import nifti_spheres_masker
from nilearn.signal import clean
from nilearn.masking import apply_mask, _unmask_3d, compute_brain_mask
from nilearn.image import load_img, new_img_like
from joblib import Parallel, delayed


def round_cust(x):
    return np.floor(x + 0.49)


def trlocked_events(events_path: str, onsets_column: str, trial_name: str,
                    bold_tr: float, bold_vols: int, separator: str = '\t'):
    """
    Loads behavior data, creates and merges into a TR (rounded -- bankers methods) dataframe to match length of BOLD.
    Trial onsets are matched to nearby TR using rounding when acquisition is not locked to TR.

    Parameters
    ----------
    events_path : str
        Path to the events data files for given subject/run.

    onsets_column : str
        Name of the column containing onset times for the event/condition.

    trial_name : str
        Name of the column containing condition/trial labels.

    bold_tr : float
        TR acquisition time (in seconds) of BOLD.

    bold_vols : int
        Number of time points for BOLD acquisition.

    separator : str, optional
        Separator used in the events data file. Default is '\t'.

    Returns
    -------
    pandas.DataFrame
        Merged dataframe with time index and events data for each event + TR delays.

    Example
    -------
    tr_locked_events = trlocked_events(events_path='./sub-01_ses-01-task-fake_events.tsv', onsets_column='OnsetTime',
    trial_name='TrialType', bold_tr=2.0, bold_vols=150)
    """
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"File '{events_path}' not found.")

    beh_df = pd.read_csv(events_path, sep=separator)

    missing_cols = [col for col in [onsets_column, trial_name] if col not in beh_df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {', '.join(missing_cols)}")

    beh_df = beh_df[[onsets_column, trial_name]]
    try:
        beh_df["TimePoint"] = round_cust(
            beh_df[onsets_column] / bold_tr).astype(int)  # Per Elizabeth, avoids bakers roundings in .round()
    except Exception as e:
        print("An error occurred:", e, "Following file included NaN, dropped.", events_path)
        beh_df.dropna(inplace=True)  # cannot perform operations on missing information
        beh_df["TimePoint"] = round_cust(beh_df[onsets_column] / bold_tr).astype(int)

    time_index = pd.RangeIndex(start=0, stop=bold_vols, step=1)
    time_index_df = pd.DataFrame(index=time_index)
    # Merge behavior data with time index
    merged_df = pd.merge(time_index_df, beh_df, how='left', left_index=True, right_on='TimePoint')

    if len(merged_df) != bold_vols:
        raise ValueError(f"Merged data length ({len(merged_df)}) doesn't match volumes ({bold_vols}).")

    return merged_df


def extract_time_series_values(behave_df: pd.DataFrame, time_series_array: np.ndarray, delay: int):
    """
    Extracts time series data from the provided time series BOLD data for associated behavioral data
    acquired from `trlocked_events` with a specified delay.

    Parameters
    ----------
    behave_df : pandas.DataFrame
        DataFrame containing behavioral data with a 'TimePoint' column indicating the starting point
        for each time series extraction.

    time_series_array : np.ndarray
        Numpy array containing time series data.

    delay : int
        Number of data points to include in each extracted time series.

    Returns
    -------
    np.ndarray
        Array containing the extracted time series data for each time point in the behavioral DataFrame.
        Each row corresponds to a time point, and each column contains the extracted time series data.


    Example
    -------
    trlocked_cuetimeseries = extract_time_series_values(behave_df, time_series_array, delay=15)
    """
    extracted_series_list = []
    for row in behave_df['TimePoint']:
        start = int(row)
        end = start + delay
        extracted_series = time_series_array[start:end]
        if len(extracted_series) < delay:  # Check if extracted series is shorter than delay
            extracted_series = np.pad(extracted_series, ((0, delay - len(extracted_series)), (0, 0)), mode='constant')
        extracted_series_list.append(extracted_series)
    return np.array(extracted_series_list, dtype=object)


def process_bold_roi_mask(bold_path: str, roi_mask: str, high_pass_sec: int = None, detrend: bool = False,
                          fwhm_smooth: float = None):
    """
        Processes BOLD data masked by a region of interest (ROI) mask file.
        Loads the BOLD and ROI mask images, applies the mask to the BOLD data, performs preprocessing (optional)
        steps including smoothing, cleaning (detrending and standardization), and averaging across time series.
        Standardizes BOLD signal using Nilearn's percent signal change ('psc')

        Parameters
        ----------
        bold_path : str
            Path to the BOLD image file.

        roi_mask : str
            Path to the ROI mask image file.

        high_pass_sec : float
            High pass filter cutoff in seconds. If None, no high pass filtering is applied.

        detrend : bool
            If True, detrend the data during cleaning.

        fwhm_smooth : float
            Full-width at half-maximum (FWHM) value for Gaussian smoothing of the BOLD data.

        Returns
        -------
        np.ndarray
            2D array containing the averaged time series data after cleaning and preprocessing.

        str
            Subject information extracted from the BOLD file name, formatted as 'sub-{sub_id}_run-{run_id}'.

        Example
        -------
        # Process BOLD data masked by ROI mask
        time_series_avg, sub_info = process_bold_roi_mask(bold_path='./sub-01_ses-01_task-fake_bold.nii.gz',
                                                         roi_mask='./siq-region_mask.nii.gz',
                                                         high_pass_sec=100.0,
                                                         detrend=True,
                                                         fwhm_smooth=5.0)
    """

    img = [load_img(i) for i in [bold_path, roi_mask]]
    bold_name = os.path.basename(bold_path)
    path_parts = bold_name.split('_')
    sub_id, run_id = None, None
    for val in path_parts:
        if 'sub-' in val:
            sub_id = val.split('-')[1]
        elif 'run-' in val:
            run_id = val.split('-')[1]
    sub_info = 'sub-' + sub_id + '_' + 'run-' + run_id

    assert img[0].shape[0:3] == img[1].shape, 'images of different shape, BOLD {} and ROI {}'.format(
        img[0].shape, img[1].shape)

    masked_data = apply_mask(bold_path, roi_mask, smoothing_fwhm=fwhm_smooth)
    clean_timeseries = clean(masked_data, standardize='psc', detrend=detrend,
                             high_pass=1 / high_pass_sec if high_pass_sec is not None else None)
    time_series_avg = np.mean(clean_timeseries, axis=1)[:, None]

    return time_series_avg, sub_info


def process_bold_roi_coords(bold_path: str, roi_mask: Nifti1Image,
                            high_pass_sec: float, detrend: bool, fwhm_smooth: float):
    """
    Processes BOLD data masked by a spherical region of interest (ROI) defined by coordinates.
    Loads the BOLD and ROI mask images, applies the spherical ROI mask to the BOLD data, performs preprocessing steps
    including smoothing, cleaning (detrending and standardization), and averaging across time series.
    Standardizes BOLD signal using Nilearn's percent signal change ('psc')

    Parameters
    ----------
    bold_path : str
        Path to the BOLD image file.

    roi_mask : nibabel.Nifti1Image
        ROI created to mask data

    high_pass_sec : float
        High pass filter cutoff in seconds. If None, no high pass filtering is applied.

    detrend : bool
        If True, detrend the data during cleaning.

    fwhm_smooth : float
        Full-width at half-maximum (FWHM) value for Gaussian smoothing of the BOLD data.

    Returns
    -------
    np.ndarray
        2D array containing the averaged time series data after cleaning and preprocessing.

    str
        Subject information extracted from the BOLD file name, formatted as 'sub-{sub_id}_run-{run_id}'.

    Example
    -------
    roi_timeseries_avg, sub_info = process_bold_roi_coords(bold_path='/path/to/bold.nii.gz',
                                                       roi_coords=(30, -15, 0),
                                                       radius_mm=5.0,
                                                       high_pass_sec=100.0,
                                                       detrend=True,
                                                       fwhm_smooth=5.0,
                                                       wb_mask='/path/to/whole_brain_mask.nii.gz')
    """
    coord_mask = roi_mask

    img = [load_img(i) for i in [bold_path, coord_mask]]
    bold_name = os.path.basename(bold_path)
    path_parts = bold_name.split('_')
    sub_id, run_id = None, None
    for val in path_parts:
        if 'sub-' in val:
            sub_id = val.split('-')[1]
        elif 'run-' in val:
            run_id = val.split('-')[1]
    sub_info = 'sub-' + sub_id + '_' + 'run-' + run_id

    assert img[0].shape[0:3] == img[1].shape, 'images of different shape, BOLD {} and ROI {}'.format(
        img[0].shape[0:3], img[1].shape)

    masked_data = apply_mask(bold_path, coord_mask, smoothing_fwhm=fwhm_smooth)
    clean_timeseries = clean(masked_data, standardize='psc', detrend=detrend,
                             high_pass=1 / high_pass_sec if high_pass_sec is not None else None)
    time_series_avg = np.mean(clean_timeseries, axis=1)[:, None]

    return time_series_avg, coord_mask, sub_info

def extract_time_series(bold_paths: list, roi_type: str, high_pass_sec: int = None, roi_mask: str = None,
                        roi_coords: tuple = None, radius_mm: int = None,
                        detrend: bool = False, fwhm_smooth: float = None, n_jobs=1):
    """
    Extracts time series data from BOLD images for specified regions of interest (ROI) or coordinates.
    For each BOLD path, extracts time series either using a mask or ROI coordinates, leveraging
    Nilearn's NiftiLabelsMasker (for mask) or nifti_spheres_masker (for coordinates).
    BOLD signal using Nilearn's percent signal change ('psc')


    Parameters
    ----------
        bold_paths : list
            List of paths to BOLD image files for subjects/runs/tasks. The order should match the order of events or
            conditions for each subject.

        roi_type : str
            Type of ROI ('mask' or 'coords').

        high_pass_sec : int, optional
            High-pass filter cutoff in seconds. If provided, converted to frequency (1/high_pass_sec). Default is None.

        roi_mask : str or None, optional
            Path to the ROI mask image. Required if roi_type is 'mask'. Default is None.

        roi_coords : tuple or None, optional
            Coordinates (x, y, z) for the center of the sphere ROI. *Required if* roi_type is 'coords'. Default is None.

        radius_mm : int or None, optional
            Radius of the sphere in millimeters. Required if roi_type is 'coords'. Default is None.

        detrend : bool, optional
            Whether to detrend the BOLD signal using Nilearn's detrend function. Default is False.

        fwhm_smooth : float or None, optional
            Full-width at half-maximum (FWHM) value for Gaussian smoothing of the BOLD data. Default is None.

        n_jobs : int, optional
            Number of CPUs to use for parallel processing. Default is 1.

    Returns
    -------
        list or tuple
            - If roi_type is 'mask':
                - List of numpy arrays containing the time series (% mean signal change) data for each subject/run.
                - List of subject information strings formatted as 'sub-{sub_id}_run-{run_id}'.
            - If roi_type is 'coords':
                - List of numpy arrays containing the averaged time series (% mean signal change) data for each subject/run.
                - Nifti1Image object representing the coordinate mask used.
                - List of subject information strings formatted as 'sub-{sub_id}_run-{run_id}'.


    Example
    -------
        # Extract percent mean signal change time series for BOLD data using a mask ROI
        roi_type = 'mask'
        bold_paths = ['./sub-01_ses-01_task-lit_bold.nii.gz', './sub-02_ses-01_task-lit_bold.nii.gz']
        roi_mask = './siq-roi_mask.nii.gz'
        time_series_list, sub_info_list = extract_time_series(bold_paths, roi_type, roi_mask=roi_mask, high_pass_sec=100, detrend=True, fwhm_smooth=5.0)

        # Extract percent mean signal change time series for BOLD data using coordinates ROI
        roi_type = 'coords'
        bold_paths = ['./sub-01_ses-01_task-lit_bold.nii.gz', './sub-02_ses_1_task-lit_bold.nii.gz']
        roi_coords = (30, -15, 0)
        time_series_list, coord_mask, sub_info_list = extract_time_series(bold_paths, roi_type, roi_coords=roi_coords, radius_mm=5, high_pass_sec=100, detrend=True, fwhm_smooth=5.0)
    """
    roi_options = ['mask', 'coords']

    if roi_type not in roi_options:
        raise ValueError("Invalid ROI type. Choose 'mask' or 'coords'.")

    if roi_type == 'mask':
        results = Parallel(n_jobs=n_jobs)(delayed(process_bold_roi_mask)(
            bold_path, roi_mask, high_pass_sec, detrend, fwhm_smooth) for bold_path in bold_paths)
        roi_series_list, id_list = zip(*results)
        return list(roi_series_list), list(id_list)

    elif roi_type == 'coords':
        # get a wb_mask
        wb_mask = compute_brain_mask(bold_paths[0])

        # create ROI
        _, roi = nifti_spheres_masker._apply_mask_and_get_affinity(
            seeds=[roi_coords], niimg=None, radius=radius_mm,
            allow_overlap=False, mask_img=wb_mask)
        coord_mask = _unmask_3d(X=roi.toarray().flatten(), mask=wb_mask.get_fdata().astype(bool))
        coord_mask = new_img_like(wb_mask, coord_mask, wb_mask.affine)

        results = Parallel(n_jobs=n_jobs)(delayed(process_bold_roi_coords)(
            bold_path, coord_mask, high_pass_sec, detrend, fwhm_smooth) for bold_path in bold_paths)
        coord_series_list, id_list = zip(*results)

        return list(coord_series_list), coord_mask, list(id_list)

    else:
        print(f'roi_type: {roi_type}, is not in [{roi_options}]')


def extract_postcue_trs_for_conditions(events_data: list, onset: str, trial_name: str,
                                       bold_tr: float, bold_vols: int, time_series: np.ndarray,
                                       conditions: list, tr_delay: int, list_trpaths: list):
    """
    Extracts time points coinciding with condition onsets plus specified delay TRs for each subjects' behavioral/timeseries data.
    Saves this information to a pandas DataFrame with associated mean signal values for each subject,
    trial and cue across the range of TRs (1 to TR-delay).

    Parameters
    ----------
    events_data : list
        List of paths to behavioral data files. Should match the order of subjects/runs/tasks as the BOLD file list.

    onset : str
        Name of the column containing onset values in the behavioral data.

    trial_name : str
        Name of the column containing condition values in the behavioral data.

    bold_tr : float
        TR (Repetition Time) for acquisition of BOLD data in seconds.

    bold_vols : int
        Number of volumes for BOLD acquisition.

    time_series : numpy.ndarray
       series_list from extract_time_series()

    conditions : list
        List of condition cues to iterate over. Must have at least one cue.

    tr_delay : int
        Number of TRs after onset of stimulus to extract and plot.

    list_trpaths : list
       id_list from extract_time_series()

    Returns
    -------
    pd.DataFrame
        DataFrame containing percent mean signal change values, subject labels, trial labels, TR values,
        and cue labels for all specified conditions.

    Example
    -------
    # Extract time points and mean signal values for conditions 'A' and 'B'
    events_dfs = ['./sub-01_ses-01_task-siq-events.csv', './sub-02_ses-01_task-siq-events.csv']
    onset = 'OnsetTime'
    trial_name = 'TrialType'
    timeseries_2subs = series list from extract_time_series()
    conditions = ['Up', 'Down']
    tr_delay = 5
    timeseries_order = id_list from extract_time_series()
    result_df = extract_postcue_trs_for_conditions(events_data=events_dfs, onset='OnsetTime', trial_name='TrialType',
    bold_tr=2.0, bold_vols=150, time_series=timeseries_2subs, conditons=['Up','Down'], tr_delay=12,
    list_trpaths=timeseries_order)
    """
    dfs = []

    # check array names first
    beh_id_list = []
    for beh_path in events_data:
        # create sub ID array to text again bold array
        beh_name = os.path.basename(beh_path)
        path_parts = beh_name.split('_')
        sub_id, run_id = None, None
        for val in path_parts:
            if 'sub-' in val:
                sub_id = val.split('-')[1]
            elif 'run-' in val:
                run_id = val.split('-')[1]
        sub_info = 'sub-' + sub_id + '_' + 'run-' + run_id
        beh_id_list.append(sub_info)

    assert len(beh_id_list) == len(list_trpaths), f"Length of behavioral files {len(beh_id_list)} " \
                                                  f"does not match TR list {len(list_trpaths)}"
    assert (np.array(beh_id_list) == np.array(list_trpaths)).all(), "Provided list_trpaths does not match" \
                                                                    f"Beh path order {beh_id_list}"

    for cue in conditions:
        cue_dfs = []  # creating separate cue dfs to accomodate different number of trials for cue types
        sub_n = 0
        for index, beh_path in enumerate(events_data):
            subset_df = trlocked_events(events_path=beh_path, onsets_column=onset,
                                        trial_name=trial_name, bold_tr=bold_tr, bold_vols=bold_vols, separator='\t')
            trial_type = subset_df[subset_df[trial_name] == cue]
            out_trs_array = extract_time_series_values(behave_df=trial_type, time_series_array=time_series[index],
                                                       delay=tr_delay)
            sub_n = sub_n + 1  # subject is equated to every event file N, subj n = 1 to len(events_data)

            # nth trial, list of TRs
            for n_trial, trs in enumerate(out_trs_array):
                num_delay = len(trs)  # Number of TRs for the current trial
                if num_delay != tr_delay:
                    raise ValueError(f"Mismatch between tr_delay ({tr_delay}) and number of delay TRs ({num_delay})")

                reshaped_array = np.array(trs).reshape(-1, 1)
                df = pd.DataFrame(reshaped_array, columns=['Mean_Signal'])
                df['Subject'] = sub_n
                df['Trial'] = n_trial + 1
                tr_values = np.arange(1, tr_delay + 1)
                df['TR'] = tr_values
                cue_values = [cue] * num_delay
                df['Cue'] = cue_values
                cue_dfs.append(df)

        dfs.append(pd.concat(cue_dfs, ignore_index=True))

    return pd.concat(dfs, ignore_index=True)


def plot_responses(df, tr: int, delay: int, style: str = 'white', save_path: str = None,
                   show_plot: bool = False, ylim: tuple = (-1, 1)):
    """
    Plots the BOLD response (Mean_Signal ~ TR) across the specified delay for cues.
    The plot uses an alpha of 0.1 with n = 1000 bootstraps for standard errors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot from extract_postcue_trs_for_conditions().
        Should include columns 'TR', 'Mean_Signal', and 'Cue'.

    tr : int
        TR value in seconds.

    delay : int
        Delay value indicating the number of TRs to plot.

    style : str, optional
        Style of the plot. Options are 'white' or 'whitegrid'. Default is 'white'.

    save_path : str, optional
        Path and filename to save the plot. If None, the plot is not saved. Default is None.

    show_plot : bool, optional
        Whether to display the plot. Default is False.

    ylim : tuple, optional
        Y-axis limits for the plot. Default is (-1, 1).


    Returns
    -------
    If show_plot = True, open backend graphics to view figure
    """
    plt.figure(figsize=(10, 8), dpi=300)
    if style not in ['white', 'whitegrid']:
        raise ValueError("Style should be white or whitegrid, provided:", style)

    sns.set(style=style, font='DejaVu Serif')

    sns.lineplot(x="TR", y="Mean_Signal", hue="Cue", style="Cue", palette="Set1",
                 errorbar='se', err_style="band", err_kws={'alpha': 0.1}, n_boot=1000,
                 legend="brief", data=df)

    # Set labels and title
    plt.xlabel(f'Seconds (TR: {tr} sec)')
    plt.ylabel('Avg. Signal Change')
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(0, delay)
    plt.xticks(np.arange(1, delay + 1, 1),
               [f'{round(i * tr, 1)}' for i in range(1, delay + 1)],
               rotation=45)

    # Show legend
    plt.legend(loc='upper right')

    # Check if save_path is provided
    if save_path:
        # Get the directory path from save_path
        directory = os.path.dirname(save_path)
        # Check if directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save plot
        plt.savefig(save_path)

    # Show plot if show_plot is True
    if not show_plot:
        plt.close()

