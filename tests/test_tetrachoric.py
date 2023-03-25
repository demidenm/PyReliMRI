from pyrelimri.tetrachoric_correlation import tetrachoric_corr
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

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
