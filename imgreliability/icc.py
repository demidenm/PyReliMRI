import warnings
from pandas import DataFrame
from numpy import sum
from scipy.stats import f
from pingouin import anova
from pingouin import intraclass_corr as pg_icc
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
        ((df_long[values].mean() - df_long[[sessions, values]].groupby(by=sessions)[values].mean())**2)*n_subjects
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

    assert icc_type in ['icc_1', 'icc_2', 'icc_3'], 'ICC type should be icc_1, icc_2,icc_3, ' \
                                                    '{} entered'.format(icc_type)

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
              sess_var: str, values: str, icc_type: str = 'icc_3'):
    """ This ICC calculation uses the SS calculation, which are similar to ANOVA, but fewer estimates are used.
    It takes in a long format pandas DF, where subjects repeat for sessions
    The total variance (SS_T) is squared difference each value and the overall mean.
    This is then decomposed into INTER (between) and INTRA (within) subject variance.

    :param df_long: Data of subjects & sessions, long format (i.e., subjs repeating, for 1+ sessions).
    :param sub_var: str of in dataframe w/ subject identifying variable
    :param sess_var: str in dataframe that is repeat session variables
    :param values: str in dataframe that contains values for each session
    :param icc_type: default is ICC(3,1), alternative is ICC(1,1) via icc_1 or ICC(2,1) via icc_2
    :return: icc calculation, icc low bound conf, icc upper bound conf, msbs, msws
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
        icc_est = (MSBtw - MSWtn) / (MSBtw + (c - 1) * MSWtn)
        icc_lb, icc_ub = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=n, n_sess=c, alpha=0.05, icc_type='icc_1')

    elif icc_type == 'icc_2':
        # ICC(2,1)
        icc_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr + c * (MSc - MSErr) / n)
        icc_lb, icc_ub = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=n, n_sess=c, icc_2=icc_est, alpha=0.05, icc_type='icc_2')

    elif icc_type == 'icc_3':
        # ICC(2,1)
        icc_est = (MSBtw - MSErr) / (MSBtw + (c - 1) * MSErr)
        icc_lb, icc_ub = icc_confint(msbs=MSBtw, msws=MSWtn, mserr=MSErr, msc=MSc,
                                     n_subjs=n, n_sess=c, alpha=0.05, icc_type='icc_3')

    return icc_est, icc_lb, icc_ub, MSBtw, MSWtn


# alternative ICC calculations that are a bit slower
def aov_icc(df_long: DataFrame, sub_var: str,
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


def peng_icc(df_long: DataFrame, sub_var: str,
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
