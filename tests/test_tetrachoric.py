from imgreliability.tetrachoric_correlation import tetrachoric_corr
import numpy as np


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
