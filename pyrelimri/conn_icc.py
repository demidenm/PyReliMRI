import os
import numpy as np
from pandas import read_csv, DataFrame
from pyrelimri.icc import sumsq_icc


def triang_to_fullmat(corr_1darray, size: int):
    """
    Convert a 1D array representing the lower triangular part of a correlation matrix (including diagonal)
    into a full NxN correlation matrix.

    Parameters
    ----------
    corr_1darray : numpy.ndarray
        A 1D array containing the elements of the lower triangular part of the correlation matrix,
        including the diagonal elements.

    size : int
        The number of variables (N), corresponding to the dimension of the resulting 2D square matrix.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the full NxN correlation matrix reconstructed from corr_1darray.
    """

    # Check if the length of the input array matches the expected length
    expected_2d = size * (size + 1) // 2
    length_1d = len(corr_1darray)

    if length_1d != expected_2d:
        raise ValueError(f"Expected length: {expected_2d}, but got {length_1d}")

    # Initialize a row-by-col matrix with zeros
    full_matrix = np.zeros((size, size))

    # Fill in the lower triangular part including the diagonal
    index = 0
    for i in range(size):
        for j in range(i + 1):
            full_matrix[i, j] = corr_1darray[index]
            index += 1

    return full_matrix


def edgewise_icc(multisession_list: list, n_cols: int, col_names: list = None,
                 separator=None, icc_type='icc_3'):
    """
    Calculates the Intraclass Correlation Coefficient (ICC), its confidence intervals (lower and upper bounds)
    and between subject, within subject and between measure variance components for each edge within specified input
    files or NDarrays using manual sum of squares calculations.
    The path to the subject's data (or ndarrays) should be provided as a list of lists for each session.

    Example of input lists for three sessions:
    dat_ses1 = ["./ses1/sub-01_ses-01_task-pilot_conn.csv", "./ses1/sub-02_ses-01_task-pilot_conn.csv",
    "./ses1/sub-03_ses-01_task-pilot_conn.csv"]
    dat_ses2 = ["./ses2/sub-01_ses-02_task-pilot_conn.csv", "./ses2/sub-02_ses-02_task-pilot_conn.csv",
    "./ses2/sub-03_ses-02_task-pilot_conn.csv"]
    dat_ses3 = ["./ses3/sub-01_ses-03_task-pilot_conn.csv", "./ses3/sub-02_ses-03_task-pilot_conn.csv",
    "./ses3/sub-03_ses-03_task-pilot_conn.csv"]

    The order of the subjects in each list must be the same.

    Two session example:
    multisession_list = [dat_ses1, dat_ses2]
    Three session example:
    multisession_list = [dat_ses1, dat_ses2, dat_ses3]

    Inter-subject variance: between subjects in sessions 1, 2, and 3
    Intra-subject variance: within subject across sessions 1, 2, and 3

    Parameters
    ----------
    multisession_list : list of lists
        Contains paths to .npy files or NDarrays for subjects' connectivity MxN square matrices for each session.

    n_cols : int
        Expected number of columns/rows in the NxN matrix.

    col_names : list of str, optional
        List of column names corresponding to the MxN matrix. Defaults to None.

    separator : str, optional
        If `multisession_list` contains file paths and not .npy extension,
        provide separator to load dataframes, e.g., ',' or '\t'. Defaults to None.

    icc_type : str, optional
        Specify ICC type. Default is 'icc_3'. Options: 'icc_1', 'icc_2', 'icc_3'.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'roi_labels': List of column names representing the connectivity edges.
            - 'est': Estimated ICCs as a 2D matrix.
            - 'lowbound': Lower bounds of ICC confidence intervals as a 2D matrix.
            - 'upbound': Upper bounds of ICC confidence intervals as a 2D matrix.
            - 'btwn_sub': Between-subject variance as a 2D matrix.
            - 'wthn_sub': Within-subject variance as a 2D matrix.
            - 'btwn_meas': Between-measure variance as a 2D matrix.

    Example
    -------
    icc_results = edgewise_icc(multisession_list=[dat_ses1, dat_ses2, dat_ses3],
                               n_cols=10, col_names=['left_pfc', 'right_pfc', ..., 'right_nacc'],
                               separator=',', icc_type='icc_3')
    """

    session_lengths = [len(session) for session in multisession_list]
    session_all_same = all(length == session_lengths[0] for length in session_lengths)

    if col_names is None:
        col_names = np.arange(1, n_cols + 1, 1)

    assert n_cols == len(
        col_names), f"Specified number ({n_cols}) of columns doesn't match " \
                    f"the length of column names ({len(col_names)})"
    assert session_all_same, f"Not all lists in session_files have the same length. " \
                             f"Mismatched lengths: {', '.join(str(length) for length in session_lengths)}"

    for i, list_set in enumerate(multisession_list):
        if all(isinstance(item, str) for item in list_set):
            print(f"All values in the list set {i} are strings")
        elif all(isinstance(item, np.ndarray) for item in list_set):
            print(f"All values in the list set {i} are ndarrays")
        else:
            raise TypeError(f"Values in the list {i} are not all NumPy ndarrays or strings. Check file types/names.")

    sub_n = np.array(multisession_list).shape[1]
    subj_list = np.arange(sub_n)
    sess_n = np.array(multisession_list).shape[0]
    corr_cols = n_cols
    sess_labels = [f"sess{i + 1}" for i in range(sess_n)]

    session_lowertriangle = []
    for session in multisession_list:
        session_vectors = []
        for matrix in session:
            if isinstance(matrix, str):
                file_extension = os.path.splitext(matrix)[1]
                # Test 5
                try:
                    if file_extension == '.npy':
                        matrix = np.load(matrix)
                    elif file_extension == '.csv':
                        matrix = read_csv(matrix, sep=separator, header=None, index_col=False).values
                    elif file_extension == '.txt':
                        matrix = np.loadtxt(matrix, delimiter=separator)
                    else:
                        print(
                            f"Unsupported file extension for file {matrix}. Supported extensions are .npy, .csv, .txt")
                except Exception as e:
                    print(f"Warning: Error loading file {matrix}: {e}")

            lower_triangle_indices_with_diag = np.tril_indices_from(matrix, k=0)
            lower_triangle_vector_with_diag = matrix[lower_triangle_indices_with_diag]
            session_vectors.append(lower_triangle_vector_with_diag)
        session_lowertriangle.append(session_vectors)

    session_lowertriangle = [np.array(session) for session in session_lowertriangle]

    est, lowbound, upbound, \
        btwn_sub, wthn_sub, btwn_meas = np.empty((6, session_lowertriangle[0].shape[-1]))

    for edge in range(session_lowertriangle[0].shape[-1]):
        np_roidata = np.column_stack((
            np.tile(subj_list, sess_n),
            np.hstack([[sess_labels[j]] * len(session_lowertriangle[j][:, edge]) for j in range(sess_n)]),
            np.hstack([session_lowertriangle[j][:, edge] for j in range(sess_n)])
        ))
        roi_pd = DataFrame(data=np_roidata, columns=["subj", "sess", "vals"])
        roi_pd = roi_pd.astype({"subj": int, "sess": "category", "vals": float})

        est[edge], lowbound[edge], upbound[edge], \
            btwn_sub[edge], wthn_sub[edge], \
            btwn_meas[edge] = sumsq_icc(df_long=roi_pd, sub_var="subj", sess_var="sess",
                                        value_var="vals", icc_type=icc_type)

    result_dict = {
        'roi_labels': col_names,
        'est': triang_to_fullmat(corr_1darray=np.array(est), size=corr_cols),
        'lowbound': triang_to_fullmat(corr_1darray=np.array(lowbound), size=corr_cols),
        'upbound': triang_to_fullmat(corr_1darray=np.array(upbound), size=corr_cols),
        'btwnsub': triang_to_fullmat(corr_1darray=np.array(btwn_sub), size=corr_cols),
        'wthnsub': triang_to_fullmat(corr_1darray=np.array(wthn_sub), size=corr_cols),
        'btwnmeas': triang_to_fullmat(corr_1darray=np.array(btwn_meas), size=corr_cols)
    }

    return result_dict
