import numpy as np
import math
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt


def crossover(snr):
    p01 = lambda x: 2*norm.cdf(-np.exp(x))
    p10 = lambda x: norm.cdf(np.exp(x) - snr) - norm.cdf(-np.exp(x) - snr)
    f_obj = lambda x: np.abs(p01(x) - p10(x))
    x_opt = opt.fmin(func=f_obj, x0=0)
    return p01(x_opt)

def upper_bound(cross_p):
    search_points = 500
    D = lambda x: x*np.log(x) - x + 1
    w = 0.5  # Setting crossover probabilities equal to each other and nu = ln(2)
    nu = np.log(2)
    xi = 0.3
    if cross_p == 0.5:
        min_ub = math.inf
    elif cross_p == 0:
        min_ub = 1
    else:
        alpha_space = np.linspace(cross_p, 1-cross_p, search_points).T
        t1 = nu*cross_p*D(alpha_space/cross_p)
        t2 = nu*cross_p*(alpha_space/w)
        t3 = nu*np.exp(-nu)*(1 - cross_p)*D(alpha_space/cross_p)
        t4 = nu*np.exp(-nu)*cross_p*D(alpha_space/cross_p)
        all_vals = np.vstack((1/t1, (1-xi)/t2, 1/t3, xi/t4))
        ub = np.max(all_vals, axis=0)
        min_ub = np.min(ub)
    return min_ub


if __name__ == "__main__":
    '''
    First plot the cross-over probabilities as a function of the SNR
    '''
    snr_space = np.logspace(-1, 1, 100)
    cross_p = [crossover(snr) for snr in snr_space]
    upper_bnd = [upper_bound(p) for p in cross_p]
    upper_bd = np.array(upper_bnd)

    plt.figure
    plt.semilogx(snr_space, cross_p)
    plt.xlabel('SNR')
    plt.ylabel('Crossover Probability')
    plt.show()

    plt.figure
    plt.loglog(snr_space, upper_bnd)
    plt.xlabel('SNR')
    plt.ylabel('C')
    plt.show()
