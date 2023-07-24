import numpy as np
from scipy.stats import f
from scipy.stats import norm
from ftest_setup import *



def get_type_I_error(n:int=30, m:int=1000, alpha:float=0.05, mu1:float=0,
                     mu2:float=0, var:float=1, roundto:int=6) -> tuple:
    """Get F test type II error

    Given:
    n - one sample size
    m - number of simulations
    alpha - significance level
    mu1 - true mean of the first sample
    mu2 - true mean of the second sample
    var - variance, true for both samples
    roundto - number of decimals numbers to round the returned values.

    Return: A tuple with values:
    [0] - empirical probability of type I error
    [1] - standard error of type I error
    """
    sd = np.sqrt(var)
    p_values = np.empty(m)

    for index in range(m):
        X1 = norm.rvs(size=n, loc=mu1, scale=sd)
        X2 = norm.rvs(size=n, loc=mu2, scale=sd)
        ftest_output = f_test(X1, X2, alpha=alpha,
                          alternative="two-sided")
        p_values[index] = ftest_output.p

    p_hat = np.mean(p_values < alpha)
    se_hat = round(np.sqrt(p_hat *(1-p_hat)/m), roundto)

    return (round(p_hat, roundto), se_hat)




def get_power(n:int=30, m:int=1000, alpha:float=0.05, mu1:float=0, mu2:float=0,
              var1:float=1, var2:float=1.5, roundto:int=6):
    """
    n - sample size
    m - number of simulations
    alpha - significance level
    mu1 - true mean of the first sample
    mu2 - true mean of the second sample
    sd1 - true standard deviation of the first sample
    sd2 - true standard deviation of the first sample
    roundto - number of decimals numbers to round the returned values.

    Return: A tuple of floats:
    [0] - power of the test
    [1] - standard error of the power
    [2] - empirical type II error probability
    """
    sd1 = np.sqrt(var1)
    sd2 = np.sqrt(var2)
    p_values = np.empty(m)

    for index in range(m):
        X1 = norm.rvs(size=n, loc=mu1, scale=sd1)
        X2 = norm.rvs(size=n, loc=mu2, scale=sd2)
        ftest_output = f_test(X1, X2, alpha=alpha, alternative="two-sided")
        p_values[index] = ftest_output.p

    power = np.mean(p_values <= 0.05)
    se_power = round(np.sqrt(power * (1-power)/m), roundto)
    type_II_error = round((1 - power), roundto)

    return round(power, roundto), se_power, type_II_error