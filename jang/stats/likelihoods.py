import numpy as np
from scipy.special import gammaln


def poisson_one_sample(nobserved: int, nbackground: float, conv: float, var: np.ndarray) -> np.ndarray:
    """Compute the likelihood Poisson(n_observed, n_background + conv * var) as a function of var."""
    nexpected = conv * var + nbackground
    loglkl = np.where(nexpected > 0, - nexpected + nobserved * np.log(nexpected) - gammaln(nobserved + 1), -np.inf)
    return np.exp(loglkl)


def poisson_several_samples(nobserved: np.ndarray, nbackground: np.ndarray, conv: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Compute the multi-sample Poisson lkl. Each argument are arrays with one entry per sample."""
    lkl = np.ones_like(var)
    for n_obs, n_bkg, cv in zip(nobserved, nbackground, conv):
        lkl *= poisson_one_sample(n_obs, n_bkg, cv, var)
    return lkl
