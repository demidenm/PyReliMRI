import numpy as np
from pandas import DataFrame
from scipy.stats import f
from numpy.typing import NDArray


def sumsq_total(df_long: DataFrame, values: str) -> NDArray:
    """
    Calculate the total sum of squares for a given column in a DataFrame.
    The total sum of squares is the sum of the squared differences between each value in the column
    and the overall mean of that column.
    Parameters
    ----------
    df_long : DataFrame
        A pandas DataFrame in long format.
    values : str
        The name of the column containing the values for which to calculate the total sum of squares.

    Returns
    -------
    NDArray
        The total sum of squares of the specified values column.

    """
    return np.sum((df_long[values] - df_long[values].mean()) ** 2)


def sumsq_within(df_long: DataFrame, sessions: str, values: str, n_subjects: int) -> NDArray:
    """
    Calculate the sum of squared within-subject variance.
    This function computes the sum of the squared differences between the average session value and the overall average
    of values, multiplied by the number of subjects.

    Parameters
    ----------
    df_long : DataFrame
        A pandas DataFrame in long format, e.g., scores across subjects and 1+ sessions.
    sessions : str
        The name of the column representing sessions (repeated measurements) in the DataFrame.
    values : str
        The name of the column containing the values for subjects across sessions.
    n_subjects : int
        The number of subjects.

    Returns
    -------
    NDArray
        The sum of squared within-subject variance.

    """

    return np.sum(
        ((df_long[values].mean() -
          df_long[[sessions, values]].groupby(by=sessions, observed=False)[values].mean()) ** 2) * n_subjects
    )


def sumsq_btwn(df_long: DataFrame, subj: str, values: str, n_sessions: int) -> NDArray:
    """
    Calculate the sum of squared between-subject variance.
    This function computes the sum of the squared differences between the average subject value and the overall average
    of values, multiplied by the number of sessions.

    Parameters
    ----------
    df_long : DataFrame
        A pandas DataFrame in long format, e.g. scores across subjects and 1+ sessions.
    subj : str
        The name of the column representing subjects (i.e. targets) in the DataFrame.
    values : str
        The name of the column containing the values for subjects (i.e. ratings) across sessions.
    n_sessions : int
        The number of sessions (i.e. raters)

    Returns
    -------
    NDArray
        The sum of squared between-subject variance.

    """
    return np.sum(
        ((df_long[values].mean() - df_long[[subj, values]].groupby(by=subj, observed=False)[values].mean()) ** 2) * n_sessions
    )


def check_icc_type(icc_type, allowed_types=None):
    if allowed_types is None:
        allowed_types = ['icc_1', 'icc_2', 'icc_3']
    assert icc_type in allowed_types, \
        f'ICC type should be in {",".join(allowed_types)}' \
        f'{icc_type} entered'


def icc_confint(msbs: float, msws: float, mserr: float, msc: float,
                n_subjs: int, n_sess: int, icc_2=None, alpha=0.05, icc_type='icc_3'):
    """
    Calculate the confidence interval for ICC(1), ICC(2,1), or ICC(3,1) using the F-distribution method.
    This function computes the 95% confidence interval for the Intraclass Correlation Coefficient (ICC) based on
    the specified ICC type (1, 2, or 3). The technique is adopted from the Pinguin library, see:
    https://pingouin-stats.org/build/html/index.html, which is based on the ICC() function from Psych package in R:
    https://www.rdocumentation.org/packages/psych/versions/2.4.3/topics/ICC

    Parameters
    ----------
    msbs : float
        The mean square between-subject.
    msws : float
        The mean square within-subject.
    mserr : float
        The mean square error.
    msc : float
        The mean square for the rater/session effect.
    n_subjs : int
        The number of subjects/targets.
    n_sess : int
        The number of sessions/raters.
    icc_2 : float, optional
        ICC(2,1) estimate used in calculating the confidence interval. Default is None.
    alpha : float, optional
        The significance level for the confidence interval. Default is 0.05.
    icc_type : str, optional
        The type of ICC for which the confidence interval is to be calculated. Default is 'icc_3'.
        Must be one of 'icc_1', 'icc_2', or 'icc_3'.

    Returns
    -------
    tuple
        The lower and upper bounds of the 95% confidence interval for the specified ICC type.
    """

    check_icc_type(icc_type)

    # Calculate F, df, and p-values
    f_stat1 = msbs / msws
    df1 = n_subjs - 1
    df1kd = n_subjs * (n_sess - 1)

    f_stat3 = msbs / mserr
    df2kd = (n_subjs - 1) * (n_sess - 1)

    # Calculate ICC Confident interval
    if icc_type == 'icc_1':
        f_lb = f_stat1 / f.ppf(1 - alpha / 2, df1, df1kd)
        f_ub = f_stat1 * f.ppf(1 - alpha / 2, df1kd, df1)
        lb_ci = (f_lb - 1) / (f_lb + (n_sess - 1))
        ub_ci = (f_ub - 1) / (f_ub + (n_sess - 1))
    elif icc_type == 'icc_2':
        fc = msc / mserr
        vn = df2kd * (n_sess * icc_2 * fc + n_subjs * (1 + (n_sess - 1) * icc_2) - n_sess * icc_2) ** 2
        vd = df1 * n_sess ** 2 * icc_2 ** 2 * fc ** 2 + (n_subjs * (1 + (n_sess - 1) * icc_2) - n_sess * icc_2) ** 2
        v = vn / vd
        f2u = f.ppf(1 - alpha / 2, n_subjs - 1, v)
        f2l = f.ppf(1 - alpha / 2, v, n_subjs - 1)
        lb_ci = n_subjs * (msbs - f2u * mserr) / (
                f2u * (n_sess * msc + (n_sess * n_subjs - n_sess - n_subjs) * mserr) + n_subjs * msbs)
        ub_ci = n_subjs * (f2l * msbs - mserr) / (
                n_sess * msc + (n_sess * n_subjs - n_sess - n_subjs) * mserr + n_subjs * f2l * msbs)
    elif icc_type == 'icc_3':
        f_lb = f_stat3 / f.ppf(1 - alpha / 2, df1, df2kd)
        f_ub = f_stat3 * f.ppf(1 - alpha / 2, df2kd, df1)
        lb_ci = (f_lb - 1) / (f_lb + (n_sess - 1))
        ub_ci = (f_ub - 1) / (f_ub + (n_sess - 1))

    return lb_ci, ub_ci


def sumsq_icc(df_long: DataFrame, sub_var: str,
              sess_var: str, value_var: str, icc_type: str = 'icc_3'):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) using the sum of squares method.
    This function calculates the ICC based on a long format DataFrame where subjects (targets) are repeated for multiple sessions (raters).
    It decomposes the total variance into total, between-subject and within-subject variance components and computes the ICC
    for the specified type (ICC(1), ICC(2,1), or ICC(3,1)).

    Parameters
    ----------
    df_long : DataFrame
        A pandas DataFrame containing the data of subjects and sessions in long format (i.e., subjects repeating for 1+ sessions).
    sub_var : str
        The column name in the DataFrame representing the subject identifier.
    sess_var : str
        The column name in the DataFrame representing the session (repeated measurement) variable.
    value_var : str
        The column name in the DataFrame containing the values for each session (rater)
    icc_type : str, optional
        The type of ICC to calculate. Default is 'icc_3'. Must be one of 'icc_1', 'icc_2', or 'icc_3'.

    Returns
    -------
    estimate : float
        The ICC estimate for the specified type.
    lowerbound : float
        The lower bound of the 95% confidence interval for the ICC estimate.
    upperbound : float
        The upper bound of the 95% confidence interval for the ICC estimate.
    btwn_sub : float
        The between-subject variance component.
    within_sub : float
        The within-subject variance component.
    btwn_measure : float, optional
        The between-measure variance component for ICC(2,1), otherwise None.
    """
    assert sub_var in df_long.columns, \
        f'sub_var {sub_var} must be a column in the data frame'
    assert sess_var in df_long.columns, \
        f'sess_var {sess_var} must be a column in the data frame'
    assert value_var in df_long.columns, \
        f'value_var {value_var} must be a column in the data frame'

    check_icc_type(icc_type)

    # check replace missing
    nan_in_vals = df_long.isna().any().any()
    if nan_in_vals:
        # Using mean based replacement; calc mean of values column
        # Note: pinguin in python & ICC in R converts data to wide --> listwise deletion --> convert to long
        mean_vals = df_long[value_var].mean()
        # Replace NaN or missing values with the column mean
        df_long[value_var].fillna(mean_vals, inplace=True)

    # num_subjs = number of subjs, num_sess = number of sessions/ratings
    num_subjs = df_long[sub_var].nunique()
    num_sess = df_long[sess_var].nunique()
    DF_r = (num_subjs - 1) * (num_sess - 1)

    # Sum of square errors
    SS_Total = sumsq_total(df_long=df_long, values=value_var)
    SS_Btw = sumsq_btwn(df_long=df_long, subj=sub_var, values=value_var, n_sessions=num_sess)
    SS_C = sumsq_within(df_long=df_long, sessions=sess_var, values=value_var, n_subjects=num_subjs)
    SS_Err = SS_Total - SS_Btw - SS_C
    SS_Wth = SS_C + SS_Err

    # Mean Sum of Squares
    MSBtw = SS_Btw / (num_subjs - 1)
    MSWtn = SS_Wth / (DF_r + (num_sess - 1))
    MSc = SS_C / (num_sess - 1)
    MSErr = SS_Err / DF_r

    # Calculate ICCs
    lowerbound, upperbound = None, None  # set to None in case they are skipped
    btwn_measure = None  # ICC(2,1) for absolute agreement includes a bias term for measures

    if icc_type == 'icc_1':
        # ICC(1), Model 1
        try:
            estimate = (MSBtw - MSWtn) / (MSBtw + (num_sess - 1) * MSWtn)
            btwn_sub = (MSBtw - MSWtn) / num_sess
            within_sub = MSWtn
        except RuntimeWarning:
            estimate = 0

        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                                 n_subjs=num_subjs, n_sess=num_sess, alpha=0.05, icc_type='icc_1')
    elif icc_type == 'icc_2':
        # ICC(2,1)
        try:
            estimate = (MSBtw - MSErr) / (MSBtw + (num_sess - 1) * MSErr + num_sess * (MSc - MSErr) / num_subjs)
            btwn_sub = (MSBtw - MSErr) / num_sess
            within_sub = MSErr
            btwn_measure = (MSc - MSErr) / num_subjs
        except RuntimeWarning:
            estimate = 0

        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                                 n_subjs=num_subjs, n_sess=num_sess, icc_2=estimate, alpha=0.05,
                                                 icc_type='icc_2')
    elif icc_type == 'icc_3':
        # ICC(3,1)
        try:
            estimate = (MSBtw - MSErr) / (MSBtw + (num_sess - 1) * MSErr)
            btwn_sub = (MSBtw - MSErr) / num_sess
            within_sub = MSErr
        except RuntimeWarning:
            estimate = 0

        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                                 n_subjs=num_subjs, n_sess=num_sess, alpha=0.05, icc_type='icc_3')

    return estimate, lowerbound, upperbound, btwn_sub, within_sub, btwn_measure
