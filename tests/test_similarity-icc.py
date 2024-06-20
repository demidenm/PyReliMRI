import pytest
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as lme
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from pyrelimri.icc import sumsq_icc
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


@pytest.mark.parametrize("rater", ['focused', 'divided'])
def test_pyreli_v_lmer_icc3(rater):

    data = sns.load_dataset('anagrams')  # load
    sub_df = data[data['attnr'] == rater]  # filter
    long_df = pd.DataFrame(
        pd.melt(sub_df,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    lmmod = lme.mixedlm("vals ~ sess", long_df, groups=long_df["subidr"], re_formula="~1")
    lmmod = lmmod.fit()
    lmmod_btwnvar = lmmod.cov_re.iloc[0, 0]
    lmmod_wthnvar = lmmod.scale
    lmmod_icc3 = lmmod_btwnvar / (lmmod_btwnvar + lmmod_wthnvar)
    icc3_test = sumsq_icc(df_long=long_df, sub_var='subidr',
                          sess_var='sess', value_var='vals', icc_type='icc_3')
    iccmod_btwnvar = icc3_test[3]
    iccmod_withinvar = icc3_test[4]
    iccmod_icc3 = icc3_test[0]

    lm_out = np.array([lmmod_btwnvar, lmmod_wthnvar, lmmod_icc3])
    pyreli_out = np.array([iccmod_btwnvar, iccmod_withinvar, iccmod_icc3])

    assert np.allclose(a=lm_out, b=pyreli_out, atol=.001)


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
                    sess_var="sess", value_var="vals", icc_type='icc_2')
    assert np.allclose(icc[0], 0.11, atol=.01)


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
