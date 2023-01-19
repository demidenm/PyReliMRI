import pandas as pd
import warnings
import numpy as np
from pingouin import anova
from pingouin import intraclass_corr as pg_icc
from numpy.typing import NDArray

def sumsq_total(df_long: pd.DataFrame, values: str) -> NDArray:
    """
    calculates the sum of square total
    the difference between each value and the global mean
    :param df_long:
    :param values: variable string for values containing scores
    :return:
    """
    return np.sum((df_long[values] - df_long[values].mean())**2)

def sumsq_within(df_long: pd.DataFrame, sessions: str, values: str, n_subjects: int) -> NDArray:
    """
    calculates the sum of squared Intra-subj variance,
    the average session value subtracted from overall avg of values
    :param df_long: long df for scores across subjects and 1+ sesssions
    :param sessions: session (repeated measurement) variable in df, string
    :param values: variable for values for subjects across sessions, str
    :param n_subjects: number of subjects
    :return: returns sumsqured within
    """

    return np.sum(
        ((df_long[values].mean() - df_long[[sessions, values]].groupby(by=sessions)[values].mean())**2)*n_subjects
    )


def sumsq_btwn(df_long: pd.DataFrame, subj: str, values: str, n_sessions: int) -> NDArray:
    """
    calculates the sum of squared between-subj variance,
    the average subject value subtracted from overall avg of values

    :param df_long: long df for scores across subjects and 1+ sesssions
    :param session: subj variable in df, string
    :param values: variable for values for subjects across sessions, str
    :param n_sessions: number of sessions
    :return: returns sumsqured within
    """
    return np.sum(
        ((df_long[values].mean() - df_long[[subj, values]].groupby(by=subj)[values].mean()) ** 2) * n_sessions
    )

def sumsq_icc(df_long: pd.DataFrame, sub_var: str,
                   sess_var: str, values: str, icc_type: str = 'icc_3'):
    """ This ICC calculation uses the SS calculation, which are similar to ANOVA, but fewer estimates are used.
    It converts a wide data.frame into a long format, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.

    :param df_long: Data of subjects & sessions, long format (i.e., subjs repeated, for 1+ sessions).
    :param sub_var: list of variables in dataframe w/ subject identifying variable
    :param sess_var: list of in dataframe that are repeat session variables
    :param values: list of values for each session
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: ICC calculation
    """
    assert icc_type in ['icc_1', 'icc_2', 'icc_3'], 'ICC type should be icc_1, icc_2,icc_3, ' \
                                                    '{} entered'.format(icc_type)

    # n = subjs, c = sessions/ratings
    n = df_long[sub_var].nunique()
    c = df_long[sess_var].nunique()
    DF_r = (n - 1) * (c - 1)

    # Sum of square errors
    SS_Total = sumsq_total(df_long=df_long, values=values)
    SS_Btw = sumsq_btwn(df_long=df_long, subj=sub_var, values=values, n_sessions=c)
    SS_C = sumsq_within(df_long=df_long, sessions=sess_var, values=values, n_subjects=n)
    SS_Err = SS_Total - SS_Btw - SS_C
    SS_Wth = SS_C + SS_Err

    # Mean Sum of Squares
    MSBtw = SS_Btw / (n - 1)
    MSWtn = SS_Wth / (DF_r + (c - 1))
    MSc = SS_C / (c - 1)
    MSErr = SS_Err / DF_r

    # Calculate ICCs
    if icc_type == 'icc_1':
        # ICC(1), Model 1
        ICC_est = (MSBtw - MSWtn) / (MSBtw + (c - 1) * MSWtn)


    elif icc_type == 'icc_2':
        # ICC(2,1)
        ICC_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr + c * (MSc - MSErr) / n)

    elif icc_type == 'icc_3':
        # ICC(2,1)
        ICC_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr)

    return ICC_est

def aov_icc(df_long: pd.DataFrame, sub_var: str,
                   sess_var: str, values: str, icc_type: str = 'icc_3'):
    """ This ICC calculation uses the ANOVA technique.
    It converts a wide data.frame into a long format, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.

    :param df_long: Data of subjects & sessions, long format (i.e., subjs repeated, for 1+ sessions).
    :param sub_var: list of variables in dataframe w/ subject identifying variable
    :param sess_var: list of in dataframe that are repeat session variables
    :param values: list of values for each session
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: ICC calculation
    """
    assert icc_type in ['icc_1', 'icc_2', 'icc_3'], 'ICC type should be icc_1, icc_2,icc_3, ' \
                                                    '{} entered'.format(icc_type)

    # n = subjs/targets, c = sessions/raters
    n = df_long[sub_var].nunique()
    c = df_long[sess_var].nunique()


    warnings.filterwarnings('ignore')
    aov = anova(data=df_long, dv=values, between=[sub_var, sess_var], ss_type=2)

    # mean of square errors calcs
    MSBtw = aov.at[0, "MS"]
    MSWtn = (aov.at[1, "SS"] + aov.at[2, "SS"]) / (aov.at[1, "DF"] + aov.at[2, "DF"])
    MSj = aov.at[1, "MS"]
    MSErr = aov.at[2, "MS"]

    # Calculate ICCs
    if icc_type == 'icc_1':
        # ICC(1), Model 1
        ICC_est = (MSBtw - MSWtn) / (MSBtw + (c - 1) * MSWtn)


    elif icc_type == 'icc_2':
        # ICC(2,1)
        ICC_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr + c * (MSj - MSErr) / n)

    elif icc_type == 'icc_3':
        # ICC(2,1)
        ICC_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr)

    return ICC_est

def peng_icc(df_long: pd.DataFrame, sub_var: str,
                   sess_var: str, values: str, icc_type: str = 'icc_3'):
    """ This ICC calculation uses the output from penguin ICC table.
    It takes in a long dataframe that consists of the sub variables, sess labels and session values

    :param df_long: Data of subjects & sessions, long format (i.e., subjs repeated, for 1+ sessions).
    :param sub_var: list of variables in dataframe w/ subject identifying variable
    :param sess_var: list of in dataframe that are repeat session variables
    :param values: list of values for each session
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: ICC calculation
    """
    assert icc_type in ['icc_1', 'icc_2', 'icc_3'], 'ICC type should be icc_1, icc_2,icc_3, ' \
                                                    '{} entered'.format(icc_type)
    warnings.filterwarnings('ignore')
    if icc_type == 'icc_1':
        # ICC(1), Model 1
        ICC_est = pg_icc(data=df_long, targets=sub_var,
                         raters=sess_var, ratings=values).iloc[0, 2]  # location of ICC 1


    elif icc_type == 'icc_2':
        # ICC(2,1)
        ICC_est = pg_icc(data=df_long, targets=sub_var,
                         raters=sess_var, ratings=values).iloc[1, 2]  # location of ICC 2

    elif icc_type == 'icc_3':
        # ICC(2,1)
        ICC_est = pg_icc(data=df_long, targets=sub_var,
                         raters=sess_var, ratings=values).iloc[2, 2]  # location of ICC 3

    return ICC_est