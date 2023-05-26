from pandas import DataFrame
from numpy import sum
from scipy.stats import f
from numpy.typing import NDArray


def sumsq_total(df_long: DataFrame, values: str) -> NDArray:
    """
    calculates the sum of square total
    the difference between each value and the global mean
    :param df_long: A pandas DataFrame in long format.
    :param values: variable string for values containing scores
    :return:
    """
    return sum((df_long[values] - df_long[values].mean())**2)


def sumsq_within(df_long: DataFrame, sessions: str, values: str, n_subjects: int) -> NDArray:
    """
    calculates the sum of squared Intra-subj variance,
    the average session value subtracted from overall avg of values
    :param df_long: A pandas DataFrame in long format (e.g scores across subjects and 1+ sessions)
    :param sessions: session (repeated measurement) variable in df, string
    :param values: variable for values for subjects across sessions, str
    :param n_subjects: number of subjects
    :return: returns sumsqured within
    """

    return sum(
        ((df_long[values].mean() - 
          df_long[[sessions, values]].groupby(by=sessions)[values].mean())**2) * n_subjects
    )


def sumsq_btwn(df_long: DataFrame, subj: str, values: str, n_sessions: int) -> NDArray:
    """
    calculates the sum of squared between-subj variance,
    the average subject value subtracted from overall avg of values

    :param df_long: A pandas DataFrame in long format (e.g scores across subjects and 1+ sessions)
    :param subj: subj variable in df, string
    :param values: variable for values for subjects across sessions, str
    :param n_sessions: number of sessions
    :return: returns sumsqured within
    """
    return sum(
        ((df_long[values].mean() - df_long[[subj, values]].groupby(by=subj)[values].mean()) ** 2) * n_sessions
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
    Calculates the confidence interval for ICC(1), ICC(2,1), or ICC(3,1) using the F-distribution method.
    Default = 95% CI. Adopts technique from Pinguin's (https://pingouin-stats.org/build/html/index.html) ICC
    confidence interval calculation

    :param msbs: The mean square between-subject (float)
    :param msws: The mean square within-subject.
    :param mserr: The mean square error.
    :param msc: The mean square for the rater effect.
    :param n_subjs: The number of subjects.
    :param n_sess: The number of raters.
    :param icc_2: ICC(2,1) estimate to be used in calculating upper/lower CI
    :param alpha: The significance level for the confidence interval. Default is 0.05.
    :param icc_type: Confidence interval for specified ICC, default ICC(3,1)

    Returns
    -------
    icc1_ci, icc21_ci or icc31_ci : tuple
        The 95% confidence intervals for ICC(1), ICC(2,1), and ICC(3,1), respectively.
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
        vn = df2kd * ((n_sess * icc_2 * fc + n_subjs * (1 + (n_sess - 1) * icc_2) - n_sess * icc_2)) ** 2
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
    """ This ICC calculation uses the SS calculation, which are similar to ANOVA, but fewer estimates are used.
    It takes in a long format pandas DF, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.

    :param df_long: Data of subjects & sessions, long format (i.e., subjs repeating, for 1+ sessions).
    :param sub_var: str of column in dataframe w/ subject identifying variable
    :param sess_var: str of column in dataframe that is repeat session variables
    :param value_var: str in dataframe that contains values for each session
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: icc calculation, icc low bound conf, icc upper bound conf, msbs, msws
    """
    assert sub_var in df_long.columns,\
        f'sub_var {sub_var} must be a column in the data frame'
    assert sess_var in df_long.columns,\
        f'sess_var {sess_var} must be a column in the data frame'
    assert value_var in df_long.columns,\
        f'value_var {value_var} must be a column in the data frame'

    check_icc_type(icc_type)

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
    if icc_type == 'icc_1':
        # ICC(1), Model 1
        estimate = (MSBtw - MSWtn) / (MSBtw + (num_sess - 1) * MSWtn)
        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=num_subjs, n_sess=num_sess, alpha=0.05, icc_type='icc_1')

    elif icc_type == 'icc_2':
        # ICC(2,1)
        estimate = (MSBtw - MSErr) / (MSBtw + (num_sess - 1) * MSErr + num_sess * (MSc - MSErr) / num_subjs)
        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=num_subjs, n_sess=num_sess, icc_2=estimate, alpha=0.05, icc_type='icc_2')

    elif icc_type == 'icc_3':
        # ICC(2,1)
        estimate = (MSBtw - MSErr) / (MSBtw + (num_sess - 1) * MSErr)
        if MSWtn > 0 and MSErr > 0:
            lowerbound, upperbound = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=num_subjs, n_sess=num_sess, alpha=0.05, icc_type='icc_3')

    return estimate, lowerbound, upperbound, MSBtw, MSWtn
