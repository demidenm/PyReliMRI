TR-by-TR Cue-locked Timeseries
================================


The `masked_timeseries` module extracts timeseries for region of interest (ROI) mask or ROI coordinates and generates TR-by-TR cue-locked timeseries. \
Below I cover three primary functions: extract_time_series extract_postcue_trs_for_conditions and plot_responses functions.
`extract_time_series` extracts the values from BOLD files, `extract_postcue_trs_for_conditions` aligns events timings to TRs ans generates TR-by-TR values \
and `plot_responses` uses the resulting values to generate a TR-by-TR figure.

**extract_time_series**

As mentioned above, the `extract_time_series` function extracts time series data from BOLD images for specified ROI or coordinates. \
This is achieve by either using NiftiLabelsMasker or nifti_spheres_masker from Nilearn to extract the timeseries and average the voxels within that ROI.

**extract_time_series**

As mentioned above, the `extract_time_series` function extracts time series data from BOLD images for specified regions of interest (ROI) or coordinates. The function uses either NiftiLabelsMasker or NiftiSpheresMasker from Nilearn to perform the extraction.

To use the `extract_time_series` function, you have to provide the following information:
    - bold_paths: A list of paths to the BOLD image files for each subject, run, and/or session.
    - roi_type: The type of ROI ('mask' or 'coords').
    - high_pass_sec: The high-pass filter cutoff in seconds (optional).
    - roi_mask: The path to the ROI mask file. If this is provided, the function will use NiftiLabelsMasker (required if roi_type is 'mask').
    - roi_coords: A tuple of coordinates (x, y, z) for the center of the sphere ROI (required if roi_type is 'coords').
    - radius_mm: The radius of the sphere in millimeters (required if roi_type is 'coords').
    - detrend: Whether to detrend the BOLD signal using Nilearn's detrend function (optional, default is False).
    - fwhm_smooth: The full-width at half-maximum (FWHM) value for Gaussian smoothing of the BOLD data (optional).
    - n_jobs: The number of CPUs to use for parallel processing (optional, default is 1). Depending on data size, at least 16GB per CPU is recommended.

The function returns:
    - If roi_type is 'mask':
        - List of numpy arrays containing the extracted time series data for each subject/run.
        - List of subject information strings formatted as 'sub-{sub_id}_run-{run_id}' reflecting the order of list of timeseries arrays.
    - If roi_type is 'coords':
        - List of numpy arrays containing the extracted time series data for each subject/run.
        - Nifti1Image coordinate mask that was used in the timeseries extraction.
        - List of subject information strings formatted as 'sub-{sub_id}_run-{run_id}' reflecting the order of list of timeseries arrays.


Example:

.. code-block:: python

    from pyrelimri import masked_timeseries

    n3_boldpaths = ["./sub-1-ses-1_task-kewl_run-01_bold.nii.gz", "./sub-2-ses-1_task-kewl_run-01_bold.nii.gz", "./sub-3-ses-1_task-kewl_run-01_bold.nii.gz"]
    roi_mask_path = "./roi_mask.nii.gz"

    # mask versus coordinates example
    timeser_mask_n3, id_order = masked_timeseries.extract_time_series(bold_paths=n3_boldpaths, roi_type='mask',
                                                                      high_pass_sec=True, roi_mask=roi_mask_path,
                                                                      detrend=True, fwhm_smooth=4, n_jobs=2)
    # Extract timeseries using ROI coordinates with a radius of 6mm
    # coordinates
    coords = [(30, -22, -18), (50, 30, 40)]
    timeser_coord_n3, roi_sphere, id_order = masked_timeseries.extract_time_series(masked_timeseries.extract_time_series(bold_paths=n3_boldpaths,
                                                                                                                      roi_type='coords', high_pass_sec=True,
                                                                                                                      roi_coords=coords, radius_mm=6,
                                                                                                                      detrend=True, fwhm_smooth=4, n_jobs=2)

**extract_postcue_trs_for_conditions**

This function extracts the TR-by-TR cue-locked timeseries for different conditions at cue onset + TR delay.

To use the `extract_postcue_trs_for_conditions` function, you have to provide the following information:
    - events_data: A list of paths to the behavioral data files. This should match the order of subjects/runs/tasks as the BOLD file list.
    - onset: The name of the column containing onset values in the behavioral data.
    - trial_name: The name of the column containing condition values in the behavioral data.
    - bold_tr: The repetition time (TR) for the acquisition of BOLD data in seconds.
    - bold_vols: The number of volumes for BOLD acquisition.
    - time_series: The timeseries data extracted using the `extract_time_series` function.
    - conditions: A list of conditions to extract the post-cue timeseries for.
    - tr_delay: The number of TRs after onset of stimulus to extract and plot.
    - list_trpaths: The list of subject information strings formatted as 'sub-{sub_id}_run-{run_id}'.

The function returns a pandas DataFrame containing mean signal intensity values, subject labels, trial labels, TR values, and cue labels for all specified conditions.

Example:

.. code-block:: python

    from pyrelimri import masked_timeseries

    # Paths to events files
    events_data = ['/sub-1-ses-1_task-kewl_run-01_events.csv', './sub-2-ses-1_task-kewl_run-01_events.csv', './sub-3-ses-1_task-kewl_run-01_events.csv']

    # Onset column name
    onset = 'onset'

    # Trial type column name for onset timees and conditions, and list of conditions to plot
    trial_name = 'trial_type'
    conditions = ['Happy', 'Sad']

    # TR delay, 0 + delay to create
    tr_delay = 5

    # Extract post-cue timeseries for conditions. Notice, timeser_mask_n3 and id_order are from above example
    out_df = masked_timeseries.extract_postcue_trs_for_conditions(
        events_data=events_data, onset=onset, trial_name=trial_name, bold_tr=2.0, bold_vols=150,
        time_series=timeser_mask_n3, conditions=conditions, tr_delay=12, list_trpaths=id_order
    )



**plot_responses**

This function plots the average response for each condition using the post-cue timeseries.

To use the `plot_responses` function, you need to provide:
    - postcue_timeseries_dict: The dictionary with post-cue timeseries for each condition.
    - conditions: The list of conditions to plot.
    - output_file: The path to save the plot image.

The function does not return any value, but it saves the plot to the specified output file.

Example:

.. code-block:: python

    # Path to save the plot image
    output_file = "./responses_plot.png"

    # Plot average responses for conditions
    masked_timeseries.plot_responses(postcue_timeseries_dict=out_df, conditions=conditions, output_file=output_file)


This will generate and save a plot of the average response for each condition to the specified output file.
