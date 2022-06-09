"""Computation of posteriors."""

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from typing import Tuple

from jang.gw import GW, get_search_region
from jang.neutrinos import Detector
from jang.parameters import Parameters
from jang.analysis import Analysis
import jang.conversions


def compute_upperlimit_from_x_y(
    x_arr: np.ndarray, y_arr: np.ndarray, CL: float = 0.9
) -> float:
    """Compute the upper limit at a confidence level CL for a given posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return np.inf
    int_arr = y_arr[:-1] * (x_arr[1:] - x_arr[:-1])
    cum_arr = np.cumsum(int_arr)
    cum_arr = 1 / cum_arr[-1] * cum_arr
    f = interp1d(cum_arr, x_arr[:-1])
    limit = float(f(CL))
    # limit dangerously closed to lkl upper bound
    if limit > 0.9 * CL * x_arr[-1]:
        return np.inf
    return limit


def normalize(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return x_arr, y_arr
    integral = np.sum(y_arr[:-1] * (x_arr[1:] - x_arr[:-1]))
    return x_arr, 1 / integral * y_arr


def poisson_one_sample(
    nobserved: int, nbackground: float, conv: float, var: np.ndarray
) -> np.ndarray:
    """Compute the likelihood Poisson(n_observed, n_background + conv * var) as a function of var."""
    nexpected = conv * var + nbackground
    return np.power(nexpected, nobserved) / factorial(nobserved) * np.exp(-nexpected)


def poisson_several_samples(
    nobserved: np.ndarray, nbackground: np.ndarray, conv: np.ndarray, var: np.ndarray,
) -> np.ndarray:
    """Compute the multi-sample Poisson lkl. Each argument are arrays with one entry per sample."""
    lkl = np.ones_like(var)
    for n_obs, n_bkg, cv in zip(nobserved, nbackground, conv):
        lkl *= poisson_one_sample(n_obs, n_bkg, cv, var)
    return lkl


def prior_signal(var: np.ndarray, bkg: np.ndarray, conv: np.ndarray, prior_type: str):

    if prior_type == "flat":
        return np.ones_like(var)
    elif prior_type == "jeffrey":
        nsamples = len(bkg)
        tmp = [
            conv[i] ** 2 / (conv[i] * var + bkg[i]) if conv[i] > 0 else 0
            for i in range(nsamples)
        ]
        return np.sqrt(np.sum(tmp, axis=0))
    else:
        raise RuntimeError(f"Unknown prior type {prior_type}")


def compute_flux_posterior(
    detector: Detector, gw: GW, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of all-flavour neutrino flux at Earth.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable flux
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)

    x_arr = np.logspace(*parameters.range_flux)
    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        post_arr += poisson_several_samples(
            toy[1].nobserved, toy[1].nbackground, phi_to_nsig, x_arr,
        ) * prior_signal(
            x_arr, toy[1].nbackground, phi_to_nsig, parameters.prior_signal,
        )
    return x_arr, post_arr


def compute_etot_posterior(
    detector: Detector, gw: GW, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of total energy.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable Etot
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn")

    x_arr = np.logspace(*parameters.range_etot)
    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        etot_to_nsig = ana.etot_to_nsig(toy)
        post_arr += poisson_several_samples(
            toy[1].nobserved, toy[1].nbackground, etot_to_nsig, x_arr,
        ) * prior_signal(
            x_arr, toy[1].nbackground, etot_to_nsig, parameters.prior_signal,
        )
    return x_arr, post_arr


def compute_fnu_posterior(
    detector: Detector, gw: GW, parameters: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of fnu=E(tot)/E(radiated).

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable fnu
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn", "radiated_energy")

    x_arr = np.logspace(*parameters.range_fnu)
    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        fnu_to_nsig = ana.fnu_to_nsig(toy)
        post_arr += poisson_several_samples(
            toy[1].nobserved, toy[1].nbackground, fnu_to_nsig, x_arr,
        ) * prior_signal(
            x_arr, toy[1].nbackground, fnu_to_nsig, parameters.prior_signal,
        )
    return x_arr, post_arr
