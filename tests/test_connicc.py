import pytest
import numpy as np
from pyrelimri.conn_icc import (triang_to_fullmat, edgewise_icc)


def test_triangtomat_valid():
    size = 3
    corr_1darray = np.array([1, 2, 3, 4, 5, 6])
    expected_output = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    output = triang_to_fullmat(corr_1darray, size)
    assert np.array_equal(output, expected_output)


def test_triangtomat_invalid():
    size = 3
    corr_1darray_invalid = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        triang_to_fullmat(corr_1darray_invalid, size)


def test_edgewise_icc_n_cols_and_col_names_length_match():
    multisession_list = [
        [np.eye(3), np.eye(3)],
        [np.eye(3), np.eye(3)]
    ]
    n_cols = 3
    col_names = ["A", "B", "C"]
    result = edgewise_icc(multisession_list, n_cols, col_names=col_names)
    assert result['roi_labels'] == col_names


def test_edgewise_icc_n_cols_and_col_names_length_mismatch():
    multisession_list = [
        [np.eye(3), np.eye(3)],
        [np.eye(3), np.eye(3)]
    ]
    n_cols = 3
    col_names_mismatch = ["A", "B"]
    with pytest.raises(AssertionError):
        edgewise_icc(multisession_list, n_cols, col_names=col_names_mismatch)


def test_edgewise_difflength():
    multisession_list = [
        [np.eye(5)],
        [np.eye(5), np.eye(5)]
    ]
    n_cols = 3
    with pytest.raises(AssertionError):
        edgewise_icc(multisession_list, n_cols)


def test_edgewise_wrongfiletype():
    multisession_list = [
        [np.eye(5), "testing.xls"],
        [np.eye(5), np.eye(5)]
    ]
    n_cols = 3
    with pytest.raises(TypeError):
        edgewise_icc(multisession_list, n_cols)


def test_edgewise_filetest_result():
    mock_matrix = np.eye(3)
    np.save('test.npy', mock_matrix)
    multisession_list_files = [
        ['test.npy', 'test.npy'],
        ['test.npy', 'test.npy']
    ]
    n_cols = 3
    result = edgewise_icc(multisession_list_files, n_cols)
    assert 'est' in result


def test_edgewise_wrong_ext():
    multisession_list_invalid_ext = [
        ['file.wrg', 'file.wrg'],
        ['file.wrg', 'file.wrg']
    ]
    n_cols = 3
    with pytest.raises(Exception):
        edgewise_icc(multisession_list_invalid_ext, n_cols)
