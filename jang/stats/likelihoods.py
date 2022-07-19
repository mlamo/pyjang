from math import factorial
import numpy as np
from scipy.special import gammaln
from typing import Dict, List

from jang.neutrinos import EventsList, Sample


def logpoisson_one_sample(nobserved: int, nbackground: float, conv: float, var: np.ndarray) -> np.ndarray:
    """Compute the likelihood Poisson(n_observed, n_background + conv * var) as a function of var."""
    nexpected = conv * var + nbackground
    loglkl = np.where(nexpected > 0, - nexpected + nobserved * np.log(nexpected) - gammaln(nobserved + 1), -np.inf)
    return loglkl


def poisson_several_samples(nobserved: np.ndarray, nbackground: np.ndarray, conv: np.ndarray, vars: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the multi-sample Poisson lkl. Each argument are arrays with one entry per sample."""
    loglkl = np.zeros_like(vars[0])
    for n_obs, n_bkg, cv in zip(nobserved, nbackground, conv):
        loglkl += logpoisson_one_sample(n_obs, n_bkg, cv, vars[0])
    return np.exp(loglkl)


def logpointsource_one_sample(sample: Sample, nobserved: int, nbackground: float,
                              conv: float, ra_src: float, dec_src: float,
                              vars: Dict[str, np.ndarray]) -> np.ndarray:
    if sample.events is None:
        return logpoisson_one_sample(nobserved, nbackground, conv, vars[0])
    nsignal = conv * vars[0]
    nexpected = nbackground + nsignal
    loglkl = np.where(nexpected > 0, - nexpected - gammaln(nobserved + 1), -np.inf)

    has_time_vars = "t0" in vars and "sigma_t" in vars

    for evt in sample.events:
        l = 0
        for n in ("signal", "background"):
            ll = locals()[f"n{n}"]
            if sample.pdfs[n]["ang"] is not None:
                if n == "signal":
                    ll *= sample.pdfs[n]["ang"](evt, ra_src, dec_src)
                else:
                    ll *= sample.pdfs[n]["ang"](evt)
            if sample.pdfs[n]["ene"] is not None:
                ll *= sample.pdfs[n]["ene"](evt)
            if sample.pdfs[n]["time"] is not None:
                if n == "signal" and has_time_vars:
                    ll *= sample.pdfs[n]["time"](evt, vars['t0'], vars['sigma_t'])
                else:
                    ll *= sample.pdfs[n]["time"](evt)
            l += ll
        loglkl += np.log(l)

    return loglkl


def pointsource_several_samples(samples: List[Sample], nobserved: np.ndarray, nbackground: np.ndarray,
                                conv: np.ndarray, ra_src: float, dec_src: float,
                                vars: Dict[str, np.ndarray]) -> np.ndarray:
    loglkl = np.zeros_like(vars[0])
    for n_obs, n_bkg, cv, s in zip(nobserved, nbackground, conv, samples):
        loglkl += logpointsource_one_sample(s, n_obs, n_bkg, cv, ra_src, dec_src, vars)
    return np.exp(loglkl)
