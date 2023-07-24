import numpy as np
from scipy.stats import f


class TestOutput(object):
    """An output of a test: statistic, p-value and confidence intervals

    Given\Attribute:
    statistic - value of the test statistic
    stat_name - name of the statistic (e.g. "F")
    dist_min - minimum value of a distribution: e.g. normal=-Infinity, F=0
    dist_max - maximum value of a distribution: e.g. +Inifinity
    p - p-value of the test
    conf_low - critical value that is the lower bound of confidence interval
    conf_max - critival value that is the upper bound of confidence interval

    """

    def __init__(self, stat: float, stat_name: str, dist_min, dist_max, p: float, conf_low=None, conf_up=None):
        self.stat_name = stat_name
        self.stat = stat
        self.p = p

        if conf_low and conf_up:
            self.ci = f"Confidence interval: ({conf_low} ; {conf_up})"
            self.rr = f"Rejection region: ({dist_min}; {conf_low}) AND ({conf_up}; {dist_max})"
        elif conf_low:
            self.ci = f"Confidence interval: ({conf_low}; {dist_max})"
            self.rr = f"Rejection region: ({dist_min}; {conf_low})"
        elif conf_up:
            self.ci = f"Confidence interval: ({dist_min}; {conf_up})"
            self.rr = f"Rejection region: ({conf_up}; {dist_max})"
        else:
            self.ci = "Cannot compute Confidence interval.\n Check if critical values are provided."
            self.rr = "Cannot compute Rejection Region.\n Check if critical values are provided."

    def __str__(self):
        return self.p_value() + '\n'+ self.statistic() + '\n' + self.ci + '\n' + self.rr + '\n'

    def statistic(self):
        """print the name and value of the test statistic
        """
        return f"{self.stat_name} statistic: {self.stat}"

    def p_value(self):
        """print the p-value of the test
        """
        return f"p-value: {self.p}"
  


def f_test(sample1, sample2, alternative: str = "two-sided", alpha: float = 0.05, roundto: int = 6):
    """F-test for two sample variance equality. Returns a dict with three values.

    Given:
    sample1, sample2 - lists or pandas.Series objects of two samples
    alternative - define the alternative hypothesis "two-sided", "greater", "less"
    alpha - significance level of the test
    roundto - number of decimals to use when rounding the number. Default is 6

    Return: a TestOutput object
    """

    x1 = np.array(sample1)
    x2 = np.array(sample2)
    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    df1 = len(x1) - 1
    df2 = len(x2) - 1

    dist_min = 0
    dist_max = "+Inifinity"
    conf_low = None
    conf_up = None

    F_stat = round(var1/var2, roundto)

    if alternative == "two-sided":
        conf_low = round(f.ppf(q=(alpha/2), dfn=df1, dfd=df2), roundto)
        conf_up = round(f.ppf(q=(1-alpha/2), dfn=df1, dfd=df2), roundto)
        p = 1 - f.cdf(x=F_stat, dfn=df1, dfd=df2)
        p_value = round(2 * min(p, 1-p), roundto)

    elif alternative == "greater":
        conf_up = round(f.ppf(1-alpha, df1, df2), roundto)
        p_value = round(1-f.cdf(x=F_stat, dfn=df1, dfd=df2), roundto)

    elif alternative == "less":
        conf_low = round(f.ppf(F_stat, df1, df2), roundto)
        p_value = round(f.cdf(x=F_stat, dfn=df1, dfd=df2), roundto)

    else:
        print("Choose an alternative: two-sided, greater or less")
        return None

    return TestOutput(F_stat,"F", dist_min=dist_min, dist_max=dist_max, p=p_value, conf_low=conf_low, conf_up=conf_up)